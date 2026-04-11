import traci
import time
traci.start(["sumo-gui", "-c", "sim.sumocfg"])

last_switch = 0
MIN_GREEN_TIME = 10  # giây

while True:
    traci.simulationStep()

    step = traci.simulation.getTime()
    time.sleep(0.1) 

    # lấy số xe từ 2 hướng
    north = traci.edge.getLastStepVehicleNumber("n2c")
    south = traci.edge.getLastStepVehicleNumber("s2c")
    east = traci.edge.getLastStepVehicleNumber("e2c")
    west  = traci.edge.getLastStepVehicleNumber("w2c")
    
    #compute total per direction
    ns_total = north + south
    ew_total = east + west
     
    # countdown calculation
    elapsed = step - last_switch
    remaining = max(0, MIN_GREEN_TIME - elapsed)

    # get current phase name
    phase = traci.trafficlight.getPhase("c")

    if phase == 0:
        state = "NS Green"
    elif phase == 2:
        state = "EW Green"
    else:
        state = "Other"

    print(f"Time {int(step)} | {state} | ⏳ {int(remaining)}s | N:{north} S:{south} E:{east} W:{west}")

    # chỉ cho đổi đèn sau 1 khoảng thời gian
    if elapsed >= MIN_GREEN_TIME:
        if ns_total > ew_total:
            new_phase = 0
        elif ew_total > ns_total:
            new_phase = 2
        else:
            new_phase = phase  # keep current if equal

        # only switch if needed
        if new_phase != phase:
            traci.trafficlight.setPhase("c", new_phase)

        last_switch = step

traci.close()