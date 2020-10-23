"""
Version: May 10th
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import Sensors

__all__ = ["Simulator", "Robot", "to_grid", "Robot_Motion", "Motion_Omni_wheel"]


def pi_shift(radians):
    # 0 to 2pi --> -pi to pi
    return (radians + math.pi) % (2 * math.pi) - math.pi


def to_grid(x, y):
    """
    :return: converted x, y
    :rtype: (int, int)
    """
    return int(np.rint(x)), int(np.rint(y))


class Robot(object):
    """
     can be used to store various values, objects, and variables
      example
      robot1 = sim.robot1
      robot1["some string identifier"] = lidar
      ...
      lidar = robot1["some string identifier"]
    """
    def __init__(self):
        self.__speed_forward = 0.0
        self.__speed_lateral = 0.0
        self.__steering = 0.0
        self.__sensors = list()
        self.__motion = Robot_Motion()
        self.__custom_values = dict()

    def set_motion_model(self, model):
        if isinstance(model, Robot_Motion):
            self.__motion = model

    def command(self, steering=0.0, speed_forward=0.0, speed_lateral=0.0):
        """
        :param steering: Wheel Steering Angle
        :type steering: float
        :param speed_forward: Speed Forward
        :type speed_forward: float
        :param speed_lateral: Speed sideways
        :type speed_lateral: float
        """
        self.__speed_forward = speed_forward
        self.__speed_lateral = speed_lateral
        self.__steering = steering

    def add_sensor(self, sensor):
        """
        :type sensor: Sensors.Sensor
        """
        if isinstance(sensor, Sensors.Sensor):
            self.__sensors.append(sensor)

    @property
    def u(self):
        """
        :return: control vector (non-measured)
        :rtype: np.ndarray
        """
        return self.__motion.get_control_vector(self.__speed_forward, self.__speed_lateral, self.command_ang_vel)

    @property
    def command_vel_forward(self):
        return self.__speed_forward

    @property
    def command_vel_lateral(self):
        return self.__speed_lateral

    @property
    def command_ang_vel(self):
        return self.__motion.steering_model(self.__steering, self.command_vel_forward, self.command_vel_lateral)

    def motion_model(self, x, y, yaw, velx, vely, ang_vel, dt):
        """
        :return: x, y, yaw
        :rtype: (float, float, float)
        """
        return self.__motion.motion_model(x=x, y=y, yaw=yaw, velx=velx, vely=vely, ang_vel=ang_vel, dt=dt)

    def store_var(self, var, key):
        """
        :param key: key used to store variable or object
        :type key: str or int
        """
        self.__custom_values[key] = var

    def retrieve_var(self, key):
        """
        Returns the variable or object stored when calling store_var()
        :param key: key used to store variable or object
        """
        if key in self.__custom_values.keys():
            return self.__custom_values[key]
        else:
            return None

    def __getitem__(self, item):
        return self.retrieve_var(key=item)

    def __setitem__(self, key, value):
        return self.store_var(var=value, key=key)

    # Called by Simulator
    def _fill_sensor_data(self, x, y, yaw, velx, vely, ang_vel, grid):
        for s in self.__sensors:
            if isinstance(s, Sensors.Sensor):
                s.generate_reading(x, y, yaw, velx, vely, ang_vel, grid)


# -------------------------------------------------------------------------
# Robot Motion: Direct Drive
# -------------------------------------------------------------------------
class Robot_Motion(object):
    def __init__(self, max_velx=3.0, max_vely=3.0, max_ang_vel=120.0):
        self._max_velx = max_velx
        self._max_vely = max_vely
        self._max_ang_vel = np.deg2rad(max_ang_vel)

    def get_control_vector(self, velx, vely, ang_vel):
        return np.asarray([velx, vely, ang_vel]).reshape(3, 1)

    def motion_model(self, x, y, yaw, velx, vely, ang_vel, dt):
        velx, vely, ang_vel = self._constrain(velx, vely, ang_vel)
        x = x + velx * math.cos(yaw) * dt
        y = y + velx * math.sin(yaw) * dt
        yaw = yaw + ang_vel * dt
        return x, y, yaw

    def steering_model(self, steering, velx, vely):
        angular_vel = 2.0 * velx * math.tan(steering)
        return angular_vel

    def _constrain(self, velx, vely, ang_vel):
        velx = max(-self._max_velx, min(velx, self._max_velx))
        vely = max(-self._max_vely, min(vely, self._max_vely))
        ang_vel = max(-self._max_ang_vel, min(ang_vel, self._max_ang_vel))
        return velx, vely, ang_vel


# -------------------------------------------------------------------------
# Robot Motion: Omni-Wheels or Mecanum Wheels
# -------------------------------------------------------------------------
class Motion_Omni_wheel(Robot_Motion):
    def steering_model(self, steering, velx, vely):
        angular_vel = 2.0 * (velx * math.tan(steering) + vely * math.tan(steering))
        return angular_vel

    def motion_model(self, x, y, yaw, velx, vely, ang_vel, dt):
        velx, vely, ang_vel = self._constrain(velx, vely, ang_vel)
        x = x + (velx * math.cos(yaw) + vely * math.cos(yaw - math.pi/2.0)) * dt
        y = y + (velx * math.sin(yaw) + vely * math.sin(yaw - math.pi/2.0)) * dt
        yaw = yaw + ang_vel * dt
        return x, y, yaw


# -------------------------------------------------------------------------
# Simulator
# -------------------------------------------------------------------------
class Simulator(object):
    __time_step = 0.1
    _window_size = (12, 6)

    def __init__(self):
        self.__ob_map = self.load_and_configure_map()
        self.__robot1 = self._Simulated_Robot(x=2, y=2, yaw=90.0, color="r")
        self.__robot2 = self._Simulated_Robot(x=23, y=2, yaw=180.0, color="b")
        fig, ax = plt.subplots(figsize=self._window_size)
        self._visual_axis = ax
        self._visual_figure = fig
        Gate1 = [[20, 1], [20, 2], [20, 3]]
        Gate2 = [[4, 23], [4, 22], [4, 21]]
        Gate3 = [[37, 1], [37, 2], [37, 3]]
        Switch1 = [[1, 21], [2, 21], [3, 21], [1, 22], [2, 22], [3, 22], [1, 23], [2, 23], [3, 23]]
        Switch2 = [[17, 12], [18, 12], [17, 13], [18, 13]]
        Switch3 = [[34, 5], [35, 5], [36, 5], [34, 6], [35, 6], [36, 6], [34, 7], [35, 7], [36, 7]]
        self.__Goal1 = [[43, 14], [43, 15], [44, 14], [44, 15]]
        self.__Goal2 = [[47, 21], [47, 22], [48, 21], [48, 22]]
        self.__switch1 = self._Switch_Gate(Switch1, Gate1, name="Switch 1")
        self.__switch2 = self._Switch_Gate(Switch2, Gate2, name="Switch 2")
        self.__switch3 = self._Switch_Gate(Switch3, Gate3, name="Switch 3")
        self.__goal1_reached = False
        self.__goal2_reached = False
        self.__goal1_markers = list()
        self.__goal2_markers = list()
        self._custom_markers = list()

    @property
    def obstacle_map(self):
        """
        :return: obstacle map
        :rtype: np.ndarray
        """
        return self.__ob_map

    @property
    def robot1(self):
        """
        :rtype: Robot
        """
        return self.__robot1.robot

    @property
    def robot2(self):
        """
        :rtype: Robot
        """
        return self.__robot2.robot

    @property
    def time_step(self):
        """
        :rtype: float
        """
        return self.__time_step

    def start_simulation(self):
        if self.__ob_map is None:
            print("Could not load obstacle map, exiting...")
            raise IOError
        self.__set_debris()
        for x in range(self.__ob_map.shape[0]):
            for y in range(self.__ob_map.shape[1]):
                if self.__ob_map[x, y] == 1:
                    self._visual_axis.plot(x, y, "sk")  # obstacle
        self.__setup_switch_visual(self.__switch1)
        self.__setup_switch_visual(self.__switch2)
        self.__setup_switch_visual(self.__switch3)
        for p in self.__Goal1:
            self.__goal1_markers.extend(self._visual_axis.plot(p[0], p[1], "og"))
        for p in self.__Goal2:
            self.__goal2_markers.extend(self._visual_axis.plot(p[0], p[1], "og"))
        self.__ob_map[self.__ob_map > 2] = 0
        self.__ob_map[self.__ob_map == 2] = 1
        self._visual_axis.set_xticks(np.arange(0, self.__ob_map.shape[0], 2))
        self._visual_axis.set_yticks(np.arange(0, self.__ob_map.shape[1], 2))
        self._visual_axis.set_xticks(np.arange(0, self.__ob_map.shape[0] + 1, 1) - 0.5, minor=True)
        self._visual_axis.set_yticks(np.arange(0, self.__ob_map.shape[1] + 1, 1) - 0.5, minor=True)
        self._visual_axis.grid(True, which='minor')
        self._visual_axis.axis("equal")
        self._visual_axis.margins(x=0.01, y=0.01)
        self.step()

    def step(self):
        # Robots
        self.__robot1.update(self.__ob_map, time_passed=self.__time_step)
        self.__robot2.update(self.__ob_map, time_passed=self.__time_step)
        self.__robot1.update_vehicle_visual()
        self.__robot2.update_vehicle_visual()
        # Switches
        if not self.__switch1.activated:
            self.__switch1.activated = self.__check_switch(self.__switch1)
        if not self.__switch2.activated:
            self.__switch2.activated = self.__check_switch(self.__switch2)
        if not self.__switch3.activated:
            self.__switch3.activated = self.__check_switch(self.__switch3)
        plt.pause(0.001)
        self.clear_markers()
        done = self.__check_goals()
        if done:
            print("Both Robots reached their goals!")
            return True
        else:
            return False

    @staticmethod
    def to_grid(x, y):
        return to_grid(x, y)

    def plot(self, x, y, color):
        marker = self._visual_axis.plot(x, y, color)
        self._custom_markers.extend(marker)

    def clear_markers(self):
        for m in self._custom_markers:
            m.remove()
        self._custom_markers = list()

    def __check_switch(self, switch):
        x1, y1 = self.__robot1.get_grid_pos()
        x2, y2 = self.__robot2.get_grid_pos()
        pos1 = [x1, y1]
        pos2 = [x2, y2]
        for p in switch.switch_cells:
            if p == pos1 or p == pos2:
                self.__unlock_gate(switch)
                return True
        return False

    def __check_goals(self):
        # Goal 1
        if not self.__goal1_reached:
            x1, y1 = self.__robot1.get_grid_pos()
            pos1 = [x1, y1]
            for p in self.__Goal1:
                if p == pos1:
                    self.__goal1_reached = True
                    print("Robot 1 reached the goal!")
                    for marker in self.__goal1_markers:
                        marker.remove()
                    break
        # Goal 2
        if not self.__goal2_reached:
            x2, y2 = self.__robot2.get_grid_pos()
            pos2 = [x2, y2]
            for p in self.__Goal2:
                if p == pos2:
                    self.__goal2_reached = True
                    print("Robot 2 reached the goal!")
                    for marker in self.__goal2_markers:
                        marker.remove()
                    break
        if self.__goal1_reached and self.__goal2_reached:
            return True
        else:
            return False

    def __unlock_gate(self, switch):
        print("{0} has been pressed! Unlocking gate...".format(switch.name))
        for p in switch.gate_cells:
            self.__ob_map[p[0], p[1]] = 0
        for marker in switch.gate_visual:
            marker.remove()
        for marker in switch.switch_visual:
            marker.remove()

    def __setup_switch_visual(self, switch):
        for p in switch.switch_cells:
            markers = self._visual_axis.plot(p[0], p[1], "oy")
            switch.switch_visual.extend(markers)
        for p in switch.gate_cells:
            markers = self._visual_axis.plot(p[0], p[1], "sb")
            switch.gate_visual.extend(markers)

    def __set_debris(self, chance=0.15):
        for x in range(self.__ob_map.shape[0]):
            for y in range(self.__ob_map.shape[1]):
                if self.__ob_map[x, y] == 4:
                    rand = random.random()
                    if rand <= chance:
                        self.__ob_map[x, y] = 1
                    else:
                        self.__ob_map[x, y] = 0

    @classmethod
    def load_and_configure_map(cls):
        grid = cls._load_map_csv(r"/Project_Obstacle_Map.csv")
        grid = cls._flip_map_horizontal(grid)
        return grid

    @staticmethod
    def _load_map_csv(filename):
        try:
            import csv
            import os
            current_dir = os.path.dirname(os.path.realpath(__file__))
            filename = current_dir + filename
            print("Loading Map: {0}".format(filename))
            with open(filename, encoding='utf-8-sig') as csvfile:
                output = list()
                reader = csv.reader(csvfile, dialect='excel')
                for row in reader:
                    output.append(row)
                output = np.asarray(output).astype(int)
                return output.T
        except IOError as e:
            print(repr(e))
            return None

    @staticmethod
    def _flip_map_horizontal(m):
        if isinstance(m, np.ndarray):
            output = np.zeros(m.shape, dtype=np.uint8)
            L = m.shape[1]
            j = L - 1
            for i in range(0, L):
                output[:, j] = m[:, i]
                j -= 1
            return output

    # -Simulated Robot Representation-
    class _Simulated_Robot(object):
        def __init__(self, x, y, yaw=0.0, color="r", vel_inaccuracy=0.05, ang_inaccuracy=0.5):
            self._speed_inaccuracy = np.diag([vel_inaccuracy, np.deg2rad(ang_inaccuracy)]) ** 2
            self.__x = x
            self.__y = y
            self.__yaw = np.deg2rad(yaw)
            self.__vel_forward = 0.0
            self.__vel_lateral = 0.0
            self.__angular_vel = 0.0
            self.robot_size = 0.7
            self.visual = None
            self.color = color
            self.robot = Robot()

        def get_grid_pos(self):
            return to_grid(self.__x, self.__y)

        def update(self, grid, time_passed=0.1):
            velx = self.robot.command_vel_forward
            vely = self.robot.command_vel_lateral
            ang_vel = self.robot.command_ang_vel
            if abs(velx) > 0:
                velx = velx + np.random.randn() * self._speed_inaccuracy[0, 0]
            if abs(vely) > 0:
                vely = vely + np.random.randn() * self._speed_inaccuracy[0, 0]
            if abs(ang_vel) > 0:
                ang_vel = ang_vel + np.random.randn() * self._speed_inaccuracy[1, 1]
            x, y, yaw = self.robot.motion_model(x=self.__x, y=self.__y, yaw=self.__yaw, velx=velx, vely=vely,
                                                ang_vel=ang_vel, dt=time_passed)
            if 0 < x < grid.shape[0] and 0 < y < grid.shape[1] and grid[to_grid(x, y)] < 1:  # check if hit obstacle
                self.__x = x
                self.__y = y
                self.__yaw = yaw
                self.__vel_forward = velx
                self.__vel_lateral = vely
                self.__angular_vel = ang_vel
            else:  # hit obstacle
                self.__vel_forward = 0.0
                self.__vel_lateral = 0.0
                self.__angular_vel = 0.0
            self.__fill_sensor_data(grid)

        def __fill_sensor_data(self, grid):
            self.robot._fill_sensor_data(self.__x, self.__y, self.__yaw, self.__vel_forward, self.__vel_lateral,
                                         self.__angular_vel, grid)

        def update_vehicle_visual(self, arrow_len=0.3):
            if self.visual is not None:
                self.visual.remove()
            self.visual = plt.arrow(self.__x, self.__y, arrow_len * math.cos(self.__yaw),
                                    arrow_len * math.sin(self.__yaw),
                                    fc=self.color, ec="k", head_width=self.robot_size, head_length=self.robot_size,
                                    head_starts_at_zero=False)

    # -Switches and Gates-
    class _Switch_Gate(object):
        def __init__(self, switch_cells, gate_cells, name="Switch"):
            self.switch_cells = switch_cells
            self.switch_visual = list()
            self.gate_cells = gate_cells
            self.gate_visual = list()
            self.activated = False
            self.name = name
