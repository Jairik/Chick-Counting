"""
Bounding box utility module for hierarchical checking logic.
In progress as we discover more bounding box scenarios.
"""

# ——— CONFIG ——————————————————————————————————————————————————
CLUSTER_THRESHOLD = 59500
OVERFIT_THRESHOLD = 12500
# —————————————————————————————————————————————————————————————

class BoundingBox:
	def __init__(self, x1, y1, x2, y2):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2

	def calculate_area(self):
		width = max(0, self.x2 - self.x1)
		height = max(0, self.y2 - self.y1)

		return width * height
	
# ——— CHECK FUNCTIONS —————————————————————————————————————————
def check_overfit(bbox):
	return 0

def check_cluster(bbox):
	return 2

def return_accurate(bbox):
	return 1

def check_and_count(bbox):
	area = bbox.calculate_area()

	if area >= CLUSTER_THRESHOLD:
		return check_cluster(bbox)
	elif area <= OVERFIT_THRESHOLD:
		return check_overfit(bbox)
	else:
		# other checks, but for now the count is accurate
		return return_accurate(bbox)

# —————————————————————————————————————————————————————————————