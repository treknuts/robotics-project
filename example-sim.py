from Sensors import *
import simulation

# simulation object
sim = simulation.Simulator()

# get robots from sim
robot1 = sim.robot1
robot2 = sim.robot2

# add sensors to robot 1
sensor1 = Debug_Sensor()  # debug sensor is ground truth, can't be used in official run
robot1.add_sensor(sensor1)

# add sensors to robot 2
sensor2 = IMU_Sensor()
sensor3 = Stereo_Camera()
robot2.add_sensor(sensor2)
robot2.add_sensor(sensor3)

# start simulation (ensure to add sensors before calling this)
sim.start_simulation()

x1 = sensor1.x
y1 = sensor1.y

# round float to int for path planning
grid_x1, grid_y1 = simulation.to_grid(x1, y1)

# get obstacle map
ob_map = sim.obstacle_map

for i in range(500):
    robot1.command(steering=0.0, speed_forward=1.0)  # send commands to robot1
    sim.step()  # call step in a loop, this updates the simulation



robot1.command()  # stops vehicle when no arguments given