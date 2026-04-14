import traci
import time

traci.start(["sumo-gui", "-c", "sim.sumocfg"])

# --- timing constants ---
MIN_GREEN_TIME = 15   # minimum green seconds before a switch is considered
YELLOW_TIME    = 3    # yellow phase duration (must match net.xml phase durations)
MAX_GREEN_TIME = 60   # force a switch after this many seconds regardless of demand

# --- state ---
last_switch   = 0
in_yellow     = False
yellow_start  = 0
pending_phase = None   # green phase to activate after yellow ends

# --- countdown POI: placed just north of the intersection centre ---
POI_ID = "countdown"
traci.poi.add(POI_ID, 100.0, 116.0, (0, 255, 0, 255),
              poiType="--", layer=10, imgFile="", width=4.0, height=4.0)

while True:
    traci.simulationStep()
    step = traci.simulation.getTime()
    time.sleep(0.05)

    phase  = traci.trafficlight.getPhase("c")
    north  = traci.edge.getLastStepVehicleNumber("n2c")
    south  = traci.edge.getLastStepVehicleNumber("s2c")
    east   = traci.edge.getLastStepVehicleNumber("e2c")
    west   = traci.edge.getLastStepVehicleNumber("w2c")
    ns_total = north + south
    ew_total = east  + west

    # ----------------------------------------------------------------
    # Yellow phase: just wait for it to expire, then activate the
    # pending green phase.
    # ----------------------------------------------------------------
    if in_yellow:
        yellow_elapsed = step - yellow_start
        remaining = max(0, YELLOW_TIME - yellow_elapsed)

        if phase == 1:
            state = "NS Yellow"
            traci.poi.setColor(POI_ID, (255, 200, 0, 255))
        else:
            state = "EW Yellow"
            traci.poi.setColor(POI_ID, (255, 200, 0, 255))

        if yellow_elapsed >= YELLOW_TIME:
            traci.trafficlight.setPhase("c", pending_phase)
            in_yellow   = False
            last_switch = step

    # ----------------------------------------------------------------
    # Green phase: count down, then decide whether to switch.
    # ----------------------------------------------------------------
    else:
        elapsed   = step - last_switch
        remaining = max(0, MIN_GREEN_TIME - elapsed)

        if phase == 0:
            state = "NS Green"
            traci.poi.setColor(POI_ID, (0, 220, 0, 255))
        elif phase == 2:
            state = "EW Green"
            traci.poi.setColor(POI_ID, (0, 220, 0, 255))
        else:
            state = "Other"

        if elapsed >= MIN_GREEN_TIME:
            # Pick the busier direction, force switch after MAX_GREEN_TIME
            if ns_total > ew_total:
                want_phase = 0
            elif ew_total > ns_total:
                want_phase = 2
            else:
                want_phase = phase   # equal demand → hold current

            if elapsed >= MAX_GREEN_TIME:
                want_phase = 2 if phase == 0 else 0  # force alternate

            if want_phase != phase:
                # Insert yellow before switching
                yellow_phase = 1 if phase == 0 else 3
                traci.trafficlight.setPhase("c", yellow_phase)
                in_yellow     = True
                yellow_start  = step
                pending_phase = want_phase
                remaining     = YELLOW_TIME
                state = "NS Yellow" if yellow_phase == 1 else "EW Yellow"
                traci.poi.setColor(POI_ID, (255, 200, 0, 255))

    # ----------------------------------------------------------------
    # Update the countdown label shown in the SUMO GUI.
    # The label text is the POI "type" string — enable "Draw poi type"
    # in SUMO-GUI View Settings > POIs to see it.
    # ----------------------------------------------------------------
    traci.poi.setType(POI_ID, f"{int(remaining)}s")

    print(
        f"Time {int(step):5d} | {state:<12} | {int(remaining):2d}s left"
        f" | N:{north:3d} S:{south:3d} E:{east:3d} W:{west:3d}"
        f" | NS:{ns_total:3d} EW:{ew_total:3d}"
    )

traci.close()
