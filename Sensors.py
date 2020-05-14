"""
Version: May 6th
"""

import numpy as np
import math

__all__ = ["Sensor", "Debug_Sensor", "IMU_Sensor", "Gyroscope", "Stereo_Camera", "Lidar", "Odometer", "Ultrasonic"]


def pi_shift(radians):
    # 0 to 2pi --> -pi to pi
    return (radians + math.pi) % (2 * math.pi) - math.pi


def to_transform_matrix(x, y, yaw):
    H = np.eye(3)
    H[0, 0] = math.cos(yaw)
    H[0, 1] = -math.sin(yaw)
    H[1, 0] = math.sin(yaw)
    H[1, 1] = math.cos(yaw)
    H[0, 2] = x
    H[1, 2] = y
    return H


class Sensor(object):

    def generate_reading(self, x, y, yaw, velx, vely, ang_vel, grid):
        pass

    def observation(self):
        pass


class Debug_Sensor(Sensor):  # omniscient, no sensor noise

    def __init__(self):
        self.__x = 0.0
        self.__y = 0.0
        self.__yaw = 0.0
        self.__vel_forward = 0.0
        self.__vel_lateral = 0.0
        self.__angular_vel = 0.0

    def generate_reading(self, x, y, yaw, velx, vely, ang_vel, grid):
        self.__x = x
        self.__y = y
        self.__yaw = yaw
        self.__vel_forward = velx
        self.__vel_lateral = vely
        self.__angular_vel = ang_vel

    def observation(self):
        return np.asarray([self.__x, self.__y, self.__yaw]).reshape(3, 1)

    @property
    def u(self):
        """
        :return: measured control vector [[velx], [vely], [ang_vel]]
        :rtype: np.ndarray
        """
        return np.asarray([self.__vel_forward, self.__vel_lateral, self.__angular_vel]).reshape(3, 1)

    @property
    def x(self):
        """
        :rtype: float
        """
        return self.__x

    @property
    def y(self):
        """
        :rtype: float
        """
        return self.__y

    @property
    def yaw(self):
        """
        :rtype: float
        """
        return self.__yaw

    @property
    def velx(self):
        """
        :rtype: float
        """
        return self.__vel_forward

    @property
    def vely(self):
        """
        :rtype: float
        """
        return self.__vel_lateral

    @property
    def ang_vel(self):
        """
        :rtype: float
        """
        return self.__angular_vel


class Gyroscope(Sensor):  # yaw, angular velocity

    def __init__(self, angle_error=0.1, ang_vel_error=0.1):
        self.noise = np.diag([np.deg2rad(angle_error), np.deg2rad(ang_vel_error)]) ** 2
        self.__yaw = 0.0
        self.__ang_vel = 0.0

    def generate_reading(self, x, y, yaw, velx, vely, ang_vel, grid):
        self.__yaw = yaw + np.random.randn() * self.noise[0, 0] ** 0.5
        self.__yaw = pi_shift(self.__yaw)
        if abs(ang_vel) > 0:
            ang_vel = ang_vel + np.random.randn() * self.noise[1, 1] ** 0.5
        self.__ang_vel = ang_vel

    def observation(self):
        return np.asarray(self.__yaw, self.__ang_vel).reshape(2, 1)

    @property
    def yaw(self):
        """
        :rtype: float
        """
        return self.__yaw

    @property
    def ang_vel(self):
        """
        :rtype: float
        """
        return self.__ang_vel


class IMU_Sensor(Sensor):  # velocities

    def __init__(self, vel_error=0.05, ang_vel_error=0.25):
        self.noise = np.diag([vel_error, np.deg2rad(ang_vel_error)]) ** 2
        self.__vel_forward = 0.0
        self.__vel_lateral = 0.0
        self.__angular_vel = 0.0

    def generate_reading(self, x, y, yaw, velx, vely, ang_vel, grid):
        if abs(velx) > 0:
            velx = velx + np.random.randn() * self.noise[0, 0] ** 0.5
        if abs(vely) > 0:
            vely = vely + np.random.randn() * self.noise[0, 0] ** 0.5
        if abs(ang_vel) > 0:
            ang_vel = ang_vel + np.random.randn() * self.noise[1, 1] ** 0.5
        self.__vel_forward = velx
        self.__vel_lateral = vely
        self.__angular_vel = ang_vel

    def observation(self):
        return np.asarray([self.__vel_forward, self.__vel_lateral, self.__angular_vel]).reshape(3, 1)

    @property
    def u(self):
        """
        :return: measured control vector [[velx], [vely], [ang_vel]]
        :rtype: np.ndarray
        """
        return np.asarray([self.__vel_forward, self.__vel_lateral, self.__angular_vel]).reshape(3, 1)

    @property
    def velx(self):
        """
        :rtype: float
        """
        return self.__vel_forward

    @property
    def vely(self):
        """
        :rtype: float
        """
        return self.__vel_lateral

    @property
    def ang_vel(self):
        """
        :rtype: float
        """
        return self.__angular_vel


class Stereo_Camera(Sensor):  # (cumulative) x, y, yaw

    def __init__(self, pos_error=0.01, angle_error=0.01):
        self.__noise = np.diag([pos_error, np.deg2rad(angle_error)])
        self.__prev_x = None
        self.__prev_y = None
        self.__prev_yaw = None
        self.__transform = to_transform_matrix(x=0.0, y=0.0, yaw=0.0)

    def generate_reading(self, x, y, yaw, velx, vely, ang_vel, grid):
        if None in [self.__prev_x, self.__prev_y, self.__prev_yaw]:
            self.__prev_x = x
            self.__prev_y = y
            self.__prev_yaw = yaw
            self.__transform = to_transform_matrix(x=x, y=y, yaw=yaw)
        else:
            dx = x - self.__prev_x
            dy = y - self.__prev_y
            dyaw = yaw - self.__prev_yaw
            if abs(dx) > 0 or abs(dy) > 0:  # check if it moved
                dyaw += np.random.randn() * self.__noise[1, 1]
                dist = math.hypot(dx, dy) * np.sign(velx + vely)
                delta = dist + np.random.randn() * self.__noise[0, 0] * dist
                dx = math.cos(dyaw) * delta
                dy = math.sin(dyaw) * delta
                self.__transform = self.__transform @ to_transform_matrix(dx, dy, dyaw)
            self.__prev_x = x
            self.__prev_y = y
            self.__prev_yaw = yaw

    def observation(self):
        """
        :rtype: np.ndarray
        """
        return np.asarray([self.x, self.y, self.yaw]).reshape(3, 1)

    @property
    def x(self):
        """
        :rtype: float
        """
        return self.__transform[0, 2]

    @property
    def y(self):
        """
        :rtype: float
        """
        return self.__transform[1, 2]

    @property
    def yaw(self):
        """
        :rtype: float
        """
        return pi_shift(math.atan2(self.__transform[1, 0], self.__transform[0, 0]))


class Lidar(Sensor):  # range observations

    def __init__(self, max_range=10.0, range_error=0.01, angle_error=0.1, angle_increment=45.0):
        self.max_range = max_range
        self.increment = angle_increment
        self.noise = np.diag([range_error, np.deg2rad(angle_error)]) ** 2
        self.__detections = list()

    def observation(self):
        return self.__detections

    @property
    def z(self):
        """
        :return: list of detections  (d, theta, id)
        :rtype: list
        """
        return self.__detections

    def generate_reading(self, x, y, yaw, velx, vely, ang_vel, grid):
        hits = self.__find_lidar_hits(x, y, grid)
        detections = list()
        for h in hits:
            dx = h[0] - x
            dy = h[1] - y
            dist = math.hypot(dx, dy)
            angle = pi_shift(math.atan2(dy, dx) - pi_shift(yaw))
            dist = dist + np.random.randn() * self.noise[0, 0] ** 0.5
            angle = angle + np.random.randn() * self.noise[1, 1] ** 0.5
            zi = [dist, pi_shift(angle), "{0},{1}".format(h[0], h[1])]
            detections.append(zi)
        self.__detections = detections

    def __find_lidar_hits(self, x, y, grid):
        hits = list()
        i = 0
        while i < 360:
            theta = np.deg2rad(i)
            x2 = x + math.cos(theta) * self.max_range
            y2 = y + math.sin(theta) * self.max_range
            start = np.asarray([x, y], dtype=np.int)
            end = np.asarray([x2, y2], dtype=np.int)
            line = self._ray_trace(start, end)
            valid_line = [p for p in line if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]]
            if valid_line:
                line = np.asarray(valid_line)
                grid_line = grid[line[:, 0], line[:, 1]]
                if np.sum(grid_line) > 0:
                    index = np.argmax(grid_line > 0)
                    detect = line[index]
                    ox = detect[0]
                    oy = detect[1]
                    hit = [ox, oy]
                    if hit not in hits:
                        hits.append(hit)
            i += self.increment
        return hits

    @staticmethod
    def _ray_trace(start, end):
        points = np.asarray([start, end])
        d0, d1 = np.abs(np.diff(points, axis=0))[0]
        if d0 > d1:
            return np.c_[np.linspace(points[0, 0], points[1, 0], d0 + 1, dtype=np.int32),
                         np.round(np.linspace(points[0, 1], points[1, 1], d0 + 1)).astype(np.int32)]
        else:
            return np.c_[np.round(np.linspace(points[0, 0], points[1, 0], d1 + 1)).astype(np.int32),
                         np.linspace(points[0, 1], points[1, 1], d1 + 1, dtype=np.int32)]


class Odometer(Sensor):  # distance traveled

    def __init__(self, dist_inaccuracy=0.01):
        self.__total_dist = 0.0
        self.__delta_dist = 0.0
        self.__prev_x = None
        self.__prev_y = None
        self.__noise = dist_inaccuracy ** 2

    def generate_reading(self, x, y, yaw, velx, vely, ang_vel, grid):
        if None in [self.__prev_x, self.__prev_y]:
            self.__prev_x = x
            self.__prev_y = y
        else:
            dx = x - self.__prev_x
            dy = y - self.__prev_y
            if abs(dx) > 0 or abs(dy) > 0:
                dist = math.hypot(dx, dy)
                self.__delta_dist = dist + np.random.randn() * self.__noise * (velx + vely)/2.0
                self.__total_dist += self.__delta_dist
            self.__prev_x = x
            self.__prev_y = y

    def observation(self):
        """
        :rtype: np.ndarray
        """
        return np.asarray([self.__delta_dist, self.__total_dist]).reshape(2, 1)

    @property
    def delta(self):
        """
        :return: distance traveled in most recent update
        :rtype: float
        """
        return self.__delta_dist

    @property
    def total(self):
        """
        :return: total distance traveled since start
        :rtype: float
        """
        return self.__total_dist


class Ultrasonic(Sensor):  # single range observation, w/ servo angle

    def __init__(self, max_range=30.0, range_error=0.01, angle_error=0.25):
        self.max_range = max_range
        self.noise = np.diag([range_error, np.deg2rad(angle_error)]) ** 2
        self.__detections = list()
        self.__servo_angle = 0.0

    def set_servo_angle(self, degrees):
        """
        :param degrees: 0 to 360
        """
        self.__servo_angle = pi_shift(np.deg2rad(degrees))

    def observation(self):
        return self.__detections

    @property
    def z(self):
        """
        :return: list of detections  (d, theta, id)
        :rtype: list
        """
        return self.observation()

    def generate_reading(self, x, y, yaw, velx, vely, ang_vel, grid):
        pulse_angle = pi_shift(yaw - self.__servo_angle)
        hits = self.__pulse(x, y, grid, theta=pulse_angle)
        detections = list()
        for h in hits:
            dx = h[0] - x
            dy = h[1] - y
            dist = math.hypot(dx, dy)
            angle = pi_shift(math.atan2(dy, dx) - pi_shift(yaw))
            dist = dist + np.random.randn() * self.noise[0, 0] ** 0.5
            angle = angle + np.random.randn() * self.noise[1, 1] ** 0.5
            zi = [dist, pi_shift(angle), "{0},{1}".format(h[0], h[1])]
            detections.append(zi)
        self.__detections = detections

    def __pulse(self, x, y, grid, theta):
        hits = list()
        x2 = x + math.cos(theta) * self.max_range
        y2 = y + math.sin(theta) * self.max_range
        start = np.asarray([x, y], dtype=np.int)
        end = np.asarray([x2, y2], dtype=np.int)
        line = self._ray_trace(start, end)
        valid_line = list()
        for p in line:
            if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]:
                valid_line.append(p)
        if valid_line:
            line = np.asarray(valid_line)
            grid_line = grid[line[:, 0], line[:, 1]]
            if np.sum(grid_line) > 0:
                index = np.argmax(grid_line > 0)
                detect = line[index]
                ox = detect[0]
                oy = detect[1]
                hits.append([ox, oy])
        return hits

    @staticmethod
    def _ray_trace(start, end):
        points = np.asarray([start, end])
        d0, d1 = np.abs(np.diff(points, axis=0))[0]
        if d0 > d1:
            return np.c_[np.linspace(points[0, 0], points[1, 0], d0 + 1, dtype=np.int32),
                         np.round(np.linspace(points[0, 1], points[1, 1], d0 + 1)).astype(np.int32)]
        else:
            return np.c_[np.round(np.linspace(points[0, 0], points[1, 0], d1 + 1)).astype(np.int32),
                         np.linspace(points[0, 1], points[1, 1], d1 + 1, dtype=np.int32)]


# more sensors to be added....