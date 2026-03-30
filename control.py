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
    east = traci.edge.getLastStepVehicleNumber("e2c")

    print(f"Time {step} | North: {north}, East: {east}")

    # chỉ cho đổi đèn sau 1 khoảng thời gian
    if step - last_switch > MIN_GREEN_TIME:

        # nếu hướng Bắc đông hơn → ưu tiên Bắc-Nam
        if north > east:
            traci.trafficlight.setPhase("c", 0)
        else:
            traci.trafficlight.setPhase("c", 2)

        last_switch = step

traci.close()