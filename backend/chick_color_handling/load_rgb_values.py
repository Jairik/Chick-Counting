"""
This script loads numpy array of rgb values, and prints a slice of entries from the array based on user command line prompt: "python load_rgb_values.py INDEX_MIN INDEX_MAX"

Used to validate numpy array contents.
"""

import sys, numpy as np

# ——— CONFIG —————————————————————————————————————————————————
NPY_PATH = ""
# ————————————————————————————————————————————————————————————

arr = np.load(NPY_PATH, mmap_mode="r")
arr_max = arr.shape[0]

if len(sys.argv) != 3:
	print("Usage: python load_rgb_values.py INDEX_MIN->int INDEX_MAX->int")
	sys.exit(1)

i_min = int(sys.argv[1])
i_max = int(sys.argv[2])

if (0 <= i_min <= i_max < arr_max):
	print(arr[i_min:i_max])
else:
	print("Max index out of range")