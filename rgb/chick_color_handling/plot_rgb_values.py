"""
This script loads numpy array of rgb values, and outputs a 3d scatterplot of said values, color coordinated.

Used to visualize numpy array contents.
"""

import numpy as np
import matplotlib.pyplot as plt

# ——— CONFIG —————————————————————————————————————————————————
NPY_PATH          = ""
MAX_PTS           = 10000  # number of entries user wants in scatterplot
# ————————————————————————————————————————————————————————————

arr = np.load(NPY_PATH, mmap_mode="r")

if len(arr) > MAX_PTS:
	idx = np.random.default_rng(0).choice(len(arr), size=MAX_PTS, replace=False)
	rgb = arr[idx]

X, Y, Z = rgb[:, 0], rgb[:, 1], rgb[:, 2]
colors = rgb.astype(float) / 255.0

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X, Y, Z, c=colors, s=1, depthshade=False)
ax.set_xlabel("red"); ax.set_ylabel("green"); ax.set_zlabel("blue")
ax.set_xlim(0, 255); ax.set_ylim(0, 255); ax.set_zlim(0, 255)
ax.set_title("RGB Distribution in Chick Bounding Box")
plt.tight_layout()
plt.show()