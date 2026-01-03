# Full fixed pipeline (copy-paste into one cell)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# --- paths (update if needed) ---
TRAIN_DIR = '/Users/hemantjangid/Desktop/Face Recognization/images/train'
TEST_DIR  = '/Users/hemantjangid/Desktop/Face Recognization/images/test'  # may not exist

# --- helper to create dataframe ---
def createdataframe(root_dir):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    image_paths, labels = [], []
    for label in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue
        for fname in sorted(os.listdir(label_path)):
            if fname.startswith('.'):
                continue
            full = os.path.join(label_path, fname)
            if os.path.isfile(full):
                image_paths.append(full)
                labels.append(label)
        print(label, "Completed")
    return image_paths, labels

# --- build train dataframe ---
train_images, train_labels = createdataframe(TRAIN_DIR)
train = pd.DataFrame({'image': train_images, 'label': train_labels})
print("Train rows:", len(train))

# --- build test dataframe only if folder exists ---
test = None
if os.path.exists(TEST_DIR):
    test_images, test_labels = createdataframe(TEST_DIR)
    test = pd.DataFrame({'image': test_images, 'label': test_labels})
    print("Test rows:", len(test))
else:
    print("TEST_DIR not found; skipping test set")

# --- feature extractor (normalizes once here) ---
def extract_features(image_paths, target_size=(48,48), color_mode='grayscale'):
    features = []
    for p in tqdm(image_paths):
        img = load_img(p, color_mode=color_mode, target_size=target_size)
        arr = img_to_array(img)            # shape (h,w,1) for grayscale
        arr = arr.astype('float32') / 255.0
        features.append(arr)
    features = np.array(features)
    return features

# --- extract features ---
train_features = extract_features(train['image'].tolist(), target_size=(48,48), color_mode='grayscale')
print("train_features dtype,shape:", train_features.dtype, train_features.shape)  # (n,48,48,1)

# --- labels: encode and one-hot (consistent) ---
classes = sorted(list(set(train['label'].tolist())))
output_class = len(classes)
print("Detected classes (count):", output_class, classes)

le = LabelEncoder()
le.fit(train['label'])
y_train_idx = le.transform(train['label'])
y_train = to_categorical(y_train_idx, num_classes=output_class)
print("y_train shape:", y_train.shape)

# --- process test if present ---
if test is not None:
    test_features = extract_features(test['image'].tolist(), target_size=(48,48), color_mode='grayscale')
    y_test_idx = le.transform(test['label'])   # use same encoder (assumes test labels are subset of train)
    y_test = to_categorical(y_test_idx, num_classes=output_class)
    print("test_features shape:", test_features.shape, "y_test shape:", y_test.shape)
else:
    test_features = None
    y_test = None

# --- sanity check: no double normalization ---
x_train = train_features   # already normalized in extractor
x_test = test_features     # if present

# --- lighter model to avoid OOM on laptop (you can increase later) ---
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_class, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- quick shape asserts to stop confusing errors ---
assert x_train.ndim == 4 and x_train.shape[1:] == (48,48,1), "x_train must be (n,48,48,1)"
assert y_train.ndim == 2 and y_train.shape[1] == output_class, "y_train shape mismatch with output_class"

# --- train (small run for test) ---
try:
    history = model.fit(
        x_train, y_train,
        batch_size=16,
        epochs=100,
        validation_split=0.1
    )
except Exception as e:
    print("Training error:", e)
    raise
def new_func():
    model_json = model.to_json()
    with open("emotiondetector.json", "w") as json_file:
        json_file.write(model_json)
    model.save("emotiondetector.h5")

new_func()
from keras.models import model_from_json
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def ef(image):
    img = load_img(image, color_mode = 'grayscale', target_size=(48,48))
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

Image = os.path.join(TRAIN_DIR, 'angry', '27.jpg')  # Update the path to use TRAIN_DIR
print("Original image of angry")
image = ef(Image)
pred = model.predict(image)  # Fix variable name from img to image
pred_label = label[pred.argmax()]
print("model predicted:", pred_label)

import matplotlib.pyplot as plt
# %matplotlib inline

image = '/Users/hemantjangid/Desktop/Face Recognization/images/train/sad/42.jpg'
print("Original image of sad")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model predicted:", pred_label)
plt.imshow(img.reshape(48,48), cmap='gray')

image = '/Users/hemantjangid/Desktop/Face Recognization/images/train/fear/2.jpg'
print("Original image of fear")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model predicted:", pred_label)
plt.imshow(img.reshape(48,48), cmap='gray')

image = '/Users/hemantjangid/Desktop/Face Recognization/images/train/disgust/299.jpg'
print("Original image of disgust")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model predicted:", pred_label)
plt.imshow(img.reshape(48,48), cmap='gray')

image = '/Users/hemantjangid/Desktop/Face Recognization/images/train/happy/7.jpg'
print("Original image of happy")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model predicted:", pred_label)
plt.imshow(img.reshape(48,48), cmap='gray')

