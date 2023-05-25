import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


dataset_folder = '/Users/rev/Desktop/Axial'

image_width, image_height = 256, 256

# Load and preprocess the dataset
images = []
labels = []

# Iterate over the class folders
class_folders = ['AD', 'CI', 'CN']
num_classes = len(class_folders)

for class_index, class_folder in enumerate(class_folders):
    class_folder_path = os.path.join(dataset_folder, class_folder)
    
    # Iterate over the images in the class folder
    for image_filename in os.listdir(class_folder_path):
        image_path = os.path.join(class_folder_path, image_filename)
        
        # Load the image and resize it to the desired dimensions
        image = load_img(image_path, target_size=(image_width, image_height))
        image_array = img_to_array(image)
        
        images.append(image_array)
        labels.append(class_index)

images = np.array(images)
labels = np.array(labels)

# Normalize the image data
images = images / 255.0
# Checking an image
sample=images[120]
plt.imshow(sample)
plt.title(label[120])

labels = to_categorical(labels)

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# Definign CNN Model parameeters
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
batch_size = 32
epochs = 10

datagen.fit(X_train)
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val))

# Evaluating the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

# Making few predictions on the test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculating evaluation metrics
precision = precision_score(np.argmax(y_test, axis=1), y_pred_labels, average='weighted')
recall = recall_score(np.argmax(y_test, axis=1), y_pred_labels, average='weighted')
f1 = f1_score(np.argmax(y_test, axis=1), y_pred_labels, average='weighted')
specificity = recall_score(np.argmax(y_test, axis=1), y_pred_labels, average='weighted', pos_label=0)
auc_roc = roc_auc_score(y_test, y_pred, average='weighted')


print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)
print('Specificity:', specificity)
print('AUC-ROC:', auc_roc)

# Plotting the accuracy in graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()
