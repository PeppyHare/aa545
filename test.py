import numpy as np


class Configuration:
    a = 1
    b = 2
    x = None

    def __init__(self):
        self.set_x()

    def set_x(self):
        self.x = 20

    def print_a(self):
        print(self.a)


class BetterConfiguration(Configuration):
    b = 30

    def set_x(self):
        self.x = -2


class Model:
    def __init__(self, c: Configuration):
        self.ab = c.b - c.a
        self.c = c

    def get_x(self):
        c: Configuration = self.c
        return c.x


m = Model(BetterConfiguration())
print(m.ab)
print(m.get_x())
print("hooray")
m2 = Model(Configuration())
print(m2.get_x())
c = Configuration()
c.print_a()