import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

t = np.array([[-1.3911 ,   1.3977 ,  -1.3766 ,   1.3699 ],
              [-1.8115 ,  -1.8115 ,   1.8115 ,   1.8115],
              [-1.0458 ,   1.0458 ,   1.0458 ,  -1.0458]])

for i in range(4):
    if i == 1:
        continue
    xs = np.array([0., t[0, i]])
    ys = np.array([0., t[1, i]])
    zs = np.array([0., t[2, i]])
    ax.plot(xs, ys, zs=zs)

plt.draw()
plt.show()
