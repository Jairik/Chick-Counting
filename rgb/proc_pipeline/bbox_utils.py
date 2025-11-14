"""
Bounding box utility module for hierarchical checking logic.
In progress as we discover more bounding box scenarios.
"""

from itertools import combinations

# ——— CONFIG ——————————————————————————————————————————————————
CLUSTER_THRESHOLD     = 59500
OVERFIT_THRESHOLD     = 12500
IOU_CONJOIN_THRESHOLD = 0.50
# —————————————————————————————————————————————————————————————

class BoundingBox:
	def __init__(self, x1, y1, x2, y2, obj_id=None):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.obj_id = obj_id

	def calculate_area(self):
		w = max(0, self.x2 - self.x1)
		h = max(0, self.y2 - self.y1)

		return w * h

	def calculate_intersect(self, other):
		xA = max(self.x1, other.x1)
		yA = max(self.y1, other.y1)
		xB = min(self.x2, other.x2)
		yB = min(self.y2, other.y2)
		w = max(0, xB - xA)
		h = max(0, yB - yA)

		return w * h

CONJOINED_SEEN = set()
CONJOINED_PARENT = {}

# ——— CHECK FUNCTIONS —————————————————————————————————————————
def bbox_key(bbox):
	if (bbox.obj_id is not None):
		return bbox.obj_id
	else:
		return (bbox.x1, bbox.y1, bbox.x2, bbox.y2)

def reset_conjoined_state():
	CONJOINED_SEEN.clear()
	CONJOINED_PARENT.clear()

def check_overfit():
	return 0

def check_cluster():
	return 2

def return_accurate():
	return 1

def check_and_count(area):
	if area >= CLUSTER_THRESHOLD:
		return check_cluster()
	elif area <= OVERFIT_THRESHOLD:
		return check_overfit()
	else:
		# other checks, but for now the count is accurate
		return return_accurate()
	
def n_way_intersection(boxes):
	x1 = max(box.x1 for box in boxes)
	y1 = max(box.y1 for box in boxes)
	x2 = min(box.x2 for box in boxes)
	y2 = min(box.y2 for box in boxes)

	w = max(0, x2 - x1)
	h = max(0, y2 - y1)

	return w * h

def combine_n_areas(boxes):
	n = len(boxes)
	total = 0

	for i in range(1, n+1):
		for subset in combinations(boxes, i):
			int_area = n_way_intersection(subset)
			if int_area == 0:
				continue
			total += int_area if (i % 2 == 1) else -int_area

	return total

def check_group(frame_boxes):
	if not frame_boxes:
		return {
			'weighted_count': 0,
			'was_conjoined': False
			}
	
	main = frame_boxes[0]
	main_key = bbox_key(main)

	if main_key in CONJOINED_PARENT:
		return {
			'weighted_count': 0,
			'was_conjoined': True,
			'conjoined_to': CONJOINED_PARENT[main_key],
			'main_id': main_key
		}

	selected_boxes = [main]
	main_area = main.calculate_area()
	child_ids = []

	for other in frame_boxes[1:]:
		k = bbox_key(other)
		if k in CONJOINED_SEEN or k in CONJOINED_PARENT:
			continue

		other_area = other.calculate_area()
		intersect = main.calculate_intersect(other)
		union = main_area + other_area - intersect
		iou = (intersect / union) if union > 0 else 0.0
		if iou >= IOU_CONJOIN_THRESHOLD:
			selected_boxes.append(other)
			CONJOINED_SEEN.add(k)
			CONJOINED_PARENT[k] = main_key
			child_ids.append(k)

	area = combine_n_areas(selected_boxes)
	wc = check_and_count(area)

	return {
		'weighted_count': wc,
		'area': area,
		'main_id': main_key,
		'conjoined_ids': child_ids,
		'was_conjoined': False
	}
