import json, os, csv, time, threading, uuid
import joblib
import numpy as np
import paho.mqtt.client as mqtt
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties
import case_config as cfg

# ============================================================
# TODO: MQTT CONFIGURATION
# ============================================================

MQTT_CONFIG = {
    "BROKER_HOST": "",
    "BROKER_PORT": 1883,
    "TOPIC_IN": "",
    "TOPIC_OUT": "",
    "CLIENT_ID": f"G48-{uuid.uuid4().hex[:8]}",
    "QOS": 1,
    "SESSION_EXPIRY_SEC": 300,
}

# ============================================================
# TODO: EDIT MODEL / OUTPUT CONFIG
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = r"model.joblib"
OUTPUT_CSV = os.path.join(HERE, "p2_prediction.csv")

#NOTE: True = Delete the existing p2_prediction.csv and start over (used during practice)
#NOTE: False = Continue from the previous file, skipping previously predicted rows (used during the actual competition)
START_NEW_CSV = True

# ============================================================
# UTILITY
# ============================================================

model = None
IDLE_TIMEOUT_SEC = 15.0
seen_timestamps = set()
last_seen_at = None
error_count = 0
connected = threading.Event()
connect_failure = {"reason": None}

# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Model type: {type(model).__name__}")

    if not hasattr(model, "predict"):
        raise RuntimeError("Loaded model does not support predict().")

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Loaded model does not support predict_proba(). "
            "Phase 2 requires probability columns."
        )
    return model

# ============================================================
# CSV SETUP
# ============================================================
def prepare_output_csv():
    global seen_timestamps

    columns = cfg.prediction_columns()

    if START_NEW_CSV:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

        seen_timestamps = set()
        print(f"Created new prediction CSV: {OUTPUT_CSV}")
        return

    if not os.path.isfile(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

        seen_timestamps = set()
        print(f"Created prediction CSV: {OUTPUT_CSV}")
        return

    with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != columns:
            raise RuntimeError(
                "Existing prediction CSV columns do not match case_config.py.\n"
                f"Expected: {columns}\n"
                f"Found:    {reader.fieldnames}"
            )

        for row in reader:
            timestamp = str(row.get("timestamp", "")).strip()
            if timestamp:
                seen_timestamps.add(timestamp)

    print(
        f"Resuming prediction CSV: {OUTPUT_CSV} "
        f"({len(seen_timestamps)} existing rows)"
    )
    if seen_timestamps:
        print("*** If these rows are from a PRACTICE run, stop now and delete the file. ***")
        print("*** Rows with the same timestamp will be skipped in this round.        ***")

def save_prediction_row(row):
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=cfg.prediction_columns(),
            extrasaction="ignore",
        )
        writer.writerow(row)

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def get_feature_values(data):
    features = {}

    for feature_name in cfg.FEATURES:
        if feature_name not in data:
            raise ValueError(f"Missing required feature '{feature_name}'")

        raw_value = data[feature_name]

        try:
            features[feature_name] = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{feature_name}' is not numeric: {raw_value}"
            )
    return features

def parse_row(payload):
    try:
        data = json.loads(payload.decode("utf-8"))
        timestamp = str(data["timestamp"]).strip()
        features = get_feature_values(data)
    except (ValueError, KeyError, TypeError):
        return None
    if not timestamp:
        return None
    return data, timestamp, features

# ============================================================
# PROBABILITY MAPPING
# ============================================================
def get_probability_values(y_prob):
    expected_class_ids = cfg.class_ids()
    probability_columns = cfg.prob_columns()

    if hasattr(model, "classes_"):
        model_class_ids = [int(c) for c in model.classes_]
    else:
        model_class_ids = expected_class_ids

    if len(y_prob) != len(model_class_ids):
        raise RuntimeError(
            "Predict_proba() output does not match model classes."
        )

    probability_by_class = {
        class_id: float(probability)
        for class_id, probability in zip(model_class_ids, y_prob)
    }

    missing_classes = [
        class_id
        for class_id in expected_class_ids
        if class_id not in probability_by_class
    ]

    if missing_classes:
        raise RuntimeError(
            f"Model has no probability output for class(es): {missing_classes}"
        )

    return {
        column_name: probability_by_class[class_id]
        for class_id, column_name
        in zip(expected_class_ids, probability_columns)
    }

# ============================================================
# MQTT CALLBACK
# ============================================================
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        connect_failure["reason"] = str(reason_code)
        print(f"Failed to connect: {reason_code}")
        connected.set()
        return

    if flags.session_present:
        print("Reconnected - session resumed, rows sent while offline will arrive now.")
    else:
        print("Connected to MQTT broker.")
    client.subscribe(MQTT_CONFIG["TOPIC_IN"], qos=MQTT_CONFIG["QOS"])
    connected.set()

def on_subscribe(client, userdata, mid, reason_code_list, properties):
    reason = reason_code_list[0]
    if reason.is_failure:
        print(f"Subscribe REJECTED for {MQTT_CONFIG['TOPIC_IN']}: {reason} - check TOPIC_IN and your team account")
    else:
        print(f"Subscribed to {MQTT_CONFIG['TOPIC_IN']} (QoS {reason.value})")

# ============================================================
# REAL-TIME PREDICTION
# ============================================================
def on_message(client, userdata, msg):
    global last_seen_at, error_count
    row = parse_row(msg.payload)
    if row is None:
        print("Skipped a message that is not a data row (does not start the round clock)")
        return
    data, timestamp, features = row

    last_seen_at = time.time()

    if timestamp in seen_timestamps:
        print(f"Duplicate timestamp skipped: {timestamp}")
        return

    try:
        x_input = np.array(
            [[features[name] for name in cfg.FEATURES]],
            dtype=float,
        )

        y_pred = int(model.predict(x_input)[0])
        y_prob = model.predict_proba(x_input)[0]

        if y_pred not in cfg.class_ids():
            raise RuntimeError(
                f"Model predicted invalid class id: {y_pred}"
            )

        y_label_pred = cfg.label_of(y_pred)
        probability_values = get_probability_values(y_prob)

        prediction_row = {
            "timestamp": timestamp,
        }

        for feature_name in cfg.FEATURES:
            prediction_row[feature_name] = features[feature_name]

        prediction_row["y_pred"] = y_pred
        prediction_row["y_label_pred"] = y_label_pred
        prediction_row.update(probability_values)

        save_prediction_row(prediction_row)
        seen_timestamps.add(timestamp)

    except PermissionError as e:
        error_count += 1
        print("=" * 60)
        print("Unable to write file - Close .csv in Excel and then run it again.")
        print(e)
        print("=" * 60)
        return

    except Exception as e:
        error_count += 1
        print("Error in on_message [Incorrect rows]:", e)
        print(">>> Number of lost rows.:", error_count)
        return

    # NOTE: Forward to your Node-RED topic.
    output_data = dict(data)
    output_data["y_pred"] = y_pred
    output_data["y_label_pred"] = y_label_pred
    output_data.update(probability_values)
    client.publish(MQTT_CONFIG["TOPIC_OUT"], json.dumps(output_data), qos=MQTT_CONFIG["QOS"])

    print(
        f"{timestamp} -> "
        f"{y_label_pred} ({y_pred}) | "
        f"Saved prediction to {OUTPUT_CSV}"
    )

# ============================================================
# MAIN
# ============================================================
def check_settings():
    blank = [
        name for name in ["BROKER_HOST", "TOPIC_IN", "TOPIC_OUT"]
        if not MQTT_CONFIG[name]
    ]

    if blank:
        raise RuntimeError(
            "Fill these in MQTT_CONFIG at the top of rt_prediction.py first: "
            + ", ".join(blank)
        )

def main():
    global model
    check_settings()
    model = load_model()
    prepare_output_csv()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"{MQTT_CONFIG['CLIENT_ID']}-rt-{uuid.uuid4().hex[:6]}",
        protocol=mqtt.MQTTv5,
    )
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message

    connect_props = Properties(PacketTypes.CONNECT)
    connect_props.SessionExpiryInterval = MQTT_CONFIG["SESSION_EXPIRY_SEC"]
    client.connect(
        MQTT_CONFIG["BROKER_HOST"],
        MQTT_CONFIG["BROKER_PORT"],
        keepalive=60,
        clean_start=mqtt.MQTT_CLEAN_START_FIRST_ONLY,
        properties=connect_props,
    )
    client.loop_start()

    if not connected.wait(15) or connect_failure["reason"]:
        client.loop_stop()
        raise RuntimeError(
            "Could not log in to the broker: %s" % (connect_failure["reason"] or "no answer")
        )

    print(f"Predictions go to {MQTT_CONFIG['TOPIC_OUT']} - use this topic in your Node-RED.")
    print("Waiting for data. The round clock starts at the first valid row. Ctrl+C to stop.")

    try:
        while True:
            if last_seen_at is not None:
                if time.time() - last_seen_at >= IDLE_TIMEOUT_SEC:
                    print("No data received %.0f seconds - Round ended."
                          % IDLE_TIMEOUT_SEC)
                    break
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping predictor...")

    finally:
        client.disconnect()
        client.loop_stop()
        print("Successfully saved:", len(seen_timestamps))
        print("Lost rows:", error_count)
        print("Disconnected from MQTT broker.")

if __name__ == "__main__":
    main()
