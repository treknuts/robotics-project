from Sensors import *
import simulation
from A_star import A_Star

sim = simulation.Simulator()

bot1 = sim.robot1
bot2 = sim.robot2

# bot1 sensors
camera1 = Stereo_Camera()
bot1.add_sensor(camera1)

# bot2 sensors
camera2 = Stereo_Camera()
bot2.add_sensor(camera2)

# Start the simulation
sim.start_simulation()

x1 = camera1.x
y1 = camera1.y

x2 = camera2.x
y2 = camera2.y

# round float to int for path planning
grid_x1, grid_y1 = simulation.to_grid(x1, y1)

ob_map = sim.obstacle_map

gate1 = A_Star(ob_map, x1, x2, ob_map[2][18], ob_map[2][18], True)
print(gate1)

for i in range(500):
    bot1.command(steering=0., speed_forward=1.)

    bot2.command(steering=0., speed_forward=1.)

    # Step sim should always be at the end of the loop
    sim.step()
