# robust_webcam_debug.py
import os, sys, traceback, json
import cv2
import numpy as np

print("PYTHON:", sys.executable, sys.version.splitlines()[0])
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print("Failed to import TensorFlow:", e)
try:
    print("cv2 version:", cv2.__version__)
except Exception as e:
    print("Failed to import cv2:", e)

# Model files - adjust names/paths if different
H5_PATH = "emotiondetector.h5"
JSON_PATH = "emotiondetector.json"
CLASSES_PATH = "classes.json"

model = None
classes = None

# Try to load classes list if present
if os.path.exists(CLASSES_PATH):
    try:
        with open(CLASSES_PATH, "r") as f:
            classes = json.load(f)
        print("Loaded classes from", CLASSES_PATH, "->", classes)
    except Exception as e:
        print("Failed to load classes.json:", e)

# Try load model (preferred: .h5)
from tensorflow.keras.models import load_model, model_from_json
try:
    if os.path.exists(H5_PATH):
        model = load_model(H5_PATH)
        print("Loaded model via load_model()", H5_PATH)
    elif os.path.exists(JSON_PATH):
        print("H5 not found, attempting JSON+weights load...")
        with open(JSON_PATH, "r") as f:
            raw = f.read()
        # sometimes JSON contains incompatible keys like 'batch_shape' -> try sanitize
        try:
            model = model_from_json(raw)
        except Exception as e_json:
            print("model_from_json raised:", e_json)
            # try sanitize JSON by removing 'batch_shape' keys
            import json as _json
            j = _json.loads(raw)
            def remove_key(obj, key="batch_shape"):
                if isinstance(obj, dict):
                    obj.pop(key, None)
                    for v in obj.values():
                        remove_key(v, key)
                elif isinstance(obj, list):
                    for it in obj:
                        remove_key(it, key)
            print(j)
            remove_key(j, "batch_shape")
            tmp = "emotiondetector_sanitized.json"
            with open(tmp, "w") as f: f.write(_json.dumps(j))
            print("Wrote sanitized JSON ->", tmp)
            model = model_from_json(_json.dumps(j))
        # if weights available, try to load
        if os.path.exists(H5_PATH):
            model.load_weights(H5_PATH)
            print("Loaded weights from", H5_PATH)
        else:
            print("Warning: loaded architecture from JSON but weights (.h5) not found.")
    else:
        raise FileNotFoundError("No model file found (emotiondetector.h5 or emotiondetector.json). Put it in the same folder.")
except Exception as e:
    print("Model load error (full traceback follows):")
    traceback.print_exc()
    sys.exit(1)

# If classes not loaded earlier, attempt using model classes from training order fallback
if classes is None:
    # fallback list — change if your training order was different
    classes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    print("Using fallback classes order:", classes)

# Print model summary and check input shape
try:
    model.summary()
    input_shape = None
    try:
        input_shape = model.input_shape  # Keras
    except:
        try:
            input_shape = model.layers[0].input_shape
        except:
            input_shape = None
    print("Model expects input shape (keras):", input_shape)
except Exception:
    print("Could not print model.summary(); continuing.")

# Make sure input shape is compatible (we'll handle common permutations)
def ensure_4d_gray(arr):
    # expects arr shape (h,w) or (h,w,1) or (1,h,w,1)
    arr = np.array(arr)
    if arr.ndim == 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
    if arr.ndim == 3:
        # add batch
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim == 4 and arr.shape[-1] not in (1,3):
        # attempt to reshape last dim
        arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
    return arr.astype('float32') / 255.0

# Haar cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
print("Haar cascade path:", haar_file)
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    print("Warning: Haar cascade failed to load from", haar_file)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Warning: could not open webcam at index 0. Try changing index to 1 or another.")
print("Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed. Exiting.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # debug print first frame faces
        # print("Detected faces:", faces)
        for (x, y, w, h) in faces:
            face_patch = gray[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            try:
                x_in = ensure_4d_gray(face_patch)
                # enforce target size if model expects specific spatial dims
                if hasattr(model, "input_shape") and model.input_shape is not None:
                    _, H, W, C = model.input_shape
                    if (H is not None and W is not None) and (x_in.shape[1] != H or x_in.shape[2] != W):
                        x_in = cv2.resize(face_patch, (W, H))
                        x_in = ensure_4d_gray(x_in)
                pred = model.predict(x_in, verbose=0)
                pred_idx = int(np.argmax(pred, axis=1)[0])
                pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)
                prob = float(np.max(pred))
                text = f"{pred_label}: {prob:.2f}"
            except Exception as ex_pred:
                print("Exception during preprocessing/predict (traceback):")
                traceback.print_exc()
                text = "pred_err"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow("Debug Emotion Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e_main:
    print("Main loop raised an exception (traceback):")
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")
