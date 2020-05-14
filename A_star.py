import numpy as np
import matplotlib.pyplot as plt


def get_id(x, y):
    return "{0},{1}".format(x, y)


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


class Node(object):

    weight = 10.0

    def __init__(self, x, y, gx, gy):
        self.x = x
        self.y = y
        self.cost = np.inf
        self.h = distance(x, y, gx, gy) * self.weight
        self.prev = None

    def f(self):
        return self.cost + self.h

    @property
    def id(self):
        return get_id(self.x, self.y)


def get_motion():
    motion = list()
    motion.append([1,0,1])
    motion.append([0,1,1])
    motion.append([-1,0,1])
    motion.append([0,-1,1])
    return np.asarray(motion)


def A_Star(grid, sx, sy, gx, gy, visual=False):
    Q = dict()
    goal = Node(gx, gy, gx, gy)
    start = Node(sx, sy, gx, gy)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0.0:
                n = Node(x, y, gx, gy)
                Q[n.id] = n
            elif visual:
                plt.plot(x, y, ".k")  # obstacle
    # --Visual--
    if visual:
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xg")
        plt.grid(True)
        plt.axis("equal")
    # ----------
    start.cost = 0.0
    Q[start.id] = start
    Q[goal.id] = goal
    motion = get_motion()
    openset = dict()
    openset[start.id] = start
    while openset:
        u = min(openset.values(), key=lambda k: k.f())
        Q.pop(u.id)
        openset.pop(u.id)
        if u.id == goal.id:  # check if goal
            return form_path(u, start)
        # --Visual--
        if visual:
            plt.plot(u.x, u.y, "xb")
            plt.pause(0.001)
        # ----------
        for i in range(len(motion)):  # loop through neighbors
            x = u.x + motion[i, 0]
            y = u.y + motion[i, 1]
            k = get_id(x, y)
            if k in Q:
                n = Q[k]
                dis = u.cost + motion[i, 2]
                if dis < n.cost:
                    n.cost = dis
                    n.prev = u
                if k not in openset:
                    openset[k] = n
                    if visual:
                        plt.plot(x, y, "xc")  # Visual
    print("NO PATH!")
    return []


def form_path(current, start):
    path = list()
    while current is not None and current.id != start.id:
        path.append(current)
        current = current.prev
    path.append(start)
    path.reverse()
    return path


if __name__ == '__main__':
    # Create Map
    ob_map = np.zeros((15, 15), dtype=int)
    ob_map[5] = 1
    ob_map[5, 8] = 0
    # Run Algorithm
    output = A_Star(ob_map, sx=0, sy=0, gx=10, gy=0, visual=True)
    # Handle Output
    for o in output:
        print(o.x, o.y)
        plt.plot(o.x, o.y, "oy") # Path Visual
    plt.show()  # Show visual