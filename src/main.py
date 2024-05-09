import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps


def lorenz(
    x: float, y: float, z: float, sigma: float = 10, r: float = 28, b: float = 8 / 3
) -> (float, float, float):
    new_x = sigma * (-x + y)
    new_y = -x * z + r * x - y
    new_z = x * y - b * z
    return new_x, new_y, new_z


dt = 0.01
steps = 10000

xs = np.empty(steps + 1, dtype="float64")
ys = np.empty(steps + 1, dtype="float64")
zs = np.empty(steps + 1, dtype="float64")

xs[0], ys[0], zs[0] = (0.0, 1.0, 1.05)

for i in range(steps):
    dot_x, dot_y, dot_z = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + dot_x * dt
    ys[i + 1] = ys[i] + dot_y * dt
    zs[i + 1] = zs[i] + dot_z * dt

cm = colormaps["RdYlBu"]
cs = colors.Normalize(vmin=0, vmax=steps)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

for i in range(steps):
    ax.plot(
        xs[i : i + 2],
        ys[i : i + 2],
        zs[i : i + 2],
        color=plt.cm.viridis((i / steps)),
        lw=0.5,
    )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()
