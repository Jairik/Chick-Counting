"""
"""

# ——— CONFIG —————————————————————————————————————————————————
CHICK_PIXELS_PATH       = ""
NON_CHICK_PIXELS_PATH   = ""
MODEL_PATH              = ""  # .joblib ext
VAL_SPLIT               = 0.20
# ————————————————————————————————————————————————————————————

import numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from chick_color_lib import make_features

def load_npy_lib(path):
	arr = np.load(path)
	return np.asarray(arr, dtype=np.uint8).reshape(-1, 3)

def build_model():
	base = Pipeline([
		("scaler", StandardScaler()),
		("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", verbose=1))
	])

	return CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)

def main():
	chick = load_npy_lib(CHICK_PIXELS_PATH)
	non_chick = load_npy_lib(NON_CHICK_PIXELS_PATH)

	X = np.vstack([chick, non_chick]).astype(np.uint8, copy=False)
	y = np.r_[np.ones(len(chick), dtype=np.uint8), np.zeros(len(non_chick), dtype=np.uint8)]

	X_train, X_val, y_train, y_val = train_test_split(
		X, y, test_size=VAL_SPLIT, stratify=y, random_state=42
	)

	model = build_model()
	model.fit(make_features(X_train), y_train)

	proba_val = model.predict_proba(make_features(X_val))[:, 1]
	auc = roc_auc_score(y_val, proba_val)
	acc = ((proba_val >= 0.5).astype(np.uint8) == y_val).mean()

	print("Training complete.")
	print(f"Validation AUC: {auc:.4f}")
	print(f"Validation ACC@0.50: {acc:.4f}")

	joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
	main()