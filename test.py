import pandas as pd
import random
from itertools import count
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

med_path = "/Users/evan/Downloads/SalesJan2009.csv"
med = pd.read_csv(med_path)
sales = pd.DataFrame(med)
ax = plt.gca()
sales.plot(kind="line", y="Latitude", ax=ax, color="red")
ax.set_xlabel("Index values")
ax.set_ylabel("Latitude values")
plt.title("Demo graph for Line plots")
plt.show()

plt.style.use("fivethirtyeight")
x_values = []
y_values = []
z_values = []
q_values = []
counter = 0
index = count()


def animate(i):

    # print(counter)

    x = next(index)  # counter or x variable -> index
    counter = next(index)
    print(counter)
    x_values.append(x)
    """
    Three random value series ->
    Y : 0-5
    Z : 3-8
    Q : 0-10
    """
    y = random.randint(0, 5)
    z = random.randint(3, 8)
    q = random.randint(0, 10)
    # append values to keep graph dynamic
    # this can be replaced with reading values from a csv files also
    # or reading values from a pandas dataframe
    y_values.append(y)
    z_values.append(z)
    q_values.append(q)

    if counter > 40:
        """
        This helps in keeping the graph fresh and refreshes values after every 40 timesteps
        """
        x_values.pop(0)
        y_values.pop(0)
        z_values.pop(0)
        q_values.pop(0)
        # counter = 0
        plt.cla()  # clears the values of the graph

    plt.plot(x_values, y_values, linestyle="--")
    plt.plot(x_values, z_values, linestyle="--")
    plt.plot(x_values, q_values, linestyle="--")

    ax.legend(["Value 1 ", "Value 2", "Value 3"])
    ax.set_xlabel("X values")
    ax.set_ylabel("Values for Three different variable")
    plt.title("Dynamic line graphs")

    time.sleep(0.25)  # keep refresh rate of 0.25 seconds


ani = FuncAnimation(plt.gcf(), animate, 1000)
plt.tight_layout()
plt.show()
