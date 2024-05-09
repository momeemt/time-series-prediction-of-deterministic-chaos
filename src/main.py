import numpy as np
import matplotlib.pyplot as plt


def lorenz(x, y, z, sigma=10, r=28, b=(8 / 3)):
    new_x = sigma * (-x + y)
    new_y = -x * z + r * x - y
    new_z = x * y - b * z
    return new_x, new_y, new_z


dt = 0.01
steps = 10000
tau = 5

xs = np.empty(steps + 1)
ys = np.empty(steps + 1)
zs = np.empty(steps + 1)

delay_xs = np.empty(steps + 1)
delay_ys = np.empty(steps + 1)
delay_zs = np.empty(steps + 1)

xs[0], ys[0], zs[0] = (0.0, 1.0, 1.05)

for i in range(steps):
    dot_x, dot_y, dot_z = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + dot_x * dt
    ys[i + 1] = ys[i] + dot_y * dt
    zs[i + 1] = zs[i] + dot_z * dt

    if i >= 2 * tau:
        delay_xs[i + 1] = xs[i + 1]
        delay_ys[i + 1] = xs[i + 1 - tau]
        delay_zs[i + 1] = xs[i + 1 - 2 * tau]

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

for i in range(steps):
    ax1.plot(
        xs[i : i + 2],
        ys[i : i + 2],
        zs[i : i + 2],
        color=plt.cm.viridis(i / steps),
        lw=0.5,
    )
    if i >= 2 * tau:
        ax2.plot(
            delay_xs[i : i + 2],
            delay_ys[i : i + 2],
            delay_zs[i : i + 2],
            color=plt.cm.viridis(i / steps),
            lw=0.5,
        )

ax1.set_title("Original Lorenz Attractor")
ax2.set_title("Delayed Coordinates")
ax1.set_xlabel("x(t)")
ax1.set_ylabel("y(t)")
ax1.set_zlabel("z(t)")
ax2.set_xlabel("x(t)")
ax2.set_ylabel("x(t-τ)")
ax2.set_zlabel("x(t-2τ)")

plt.show()
