# ============================================================
# CONFIGURATION
# ============================================================
CASE_ID = "c.grh"
CASE_TITLE = "Greenhouse Environment State"

MODEL_NAME = "decision_tree"
FEATURES = ["temp_c", "humidity_pct", "lux_lm", "co2_ppm"]
CLASS_LABELS = ["normal", "alert", "alarm"]

# ============================================================
# UTILITY
# ============================================================
def label_of(class_id):
    return CLASS_LABELS[int(class_id)]

def class_ids():
    return list(range(len(CLASS_LABELS)))

def prob_columns():
    return ["y_pred_prob_" + label for label in CLASS_LABELS]

def feature_columns():
    return ["timestamp"] + FEATURES

def truth_columns():
    return ["timestamp"] + FEATURES + ["y", "y_label"]

def prediction_columns():
    return ["timestamp"] + FEATURES + ["y_pred", "y_label_pred"] + prob_columns()
