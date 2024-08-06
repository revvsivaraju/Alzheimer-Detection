import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA

# Dataset and image parameters
dataset_folder = '/Users/rev/Desktop/Axial'
image_width, image_height = 256, 256

# Load and preprocess the dataset
images = []
labels = []
class_folders = ['AD', 'CI', 'CN']
num_classes = len(class_folders)

for class_index, class_folder in enumerate(class_folders):
    class_folder_path = os.path.join(dataset_folder, class_folder)
    
    for image_filename in os.listdir(class_folder_path):
        image_path = os.path.join(class_folder_path, image_filename)
        
        image = load_img(image_path, target_size=(image_width, image_height))
        image_array = img_to_array(image)
        
        images.append(image_array)
        labels.append(class_index)

images = np.array(images)
labels = np.array(labels)

# Normalize the image data
images = images / 255.0

# Convert labels to categorical
labels = to_categorical(labels)

# Reshape images for PCA
n_samples, nx, ny, nz = images.shape
images_2d = images.reshape((n_samples, nx*ny*nz))

# Apply PCA
n_components = 100  # You can adjust this number
pca = PCA(n_components=n_components)
images_pca = pca.fit_transform(images_2d)

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_pca, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(n_components,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training parameters
batch_size = 32
epochs = 10

# Train the model
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate evaluation metrics
precision = precision_score(np.argmax(y_test, axis=1), y_pred_labels, average='weighted')
recall = recall_score(np.argmax(y_test, axis=1), y_pred_labels, average='weighted')
f1 = f1_score(np.argmax(y_test, axis=1), y_pred_labels, average='weighted')
specificity = recall_score(np.argmax(y_test, axis=1), y_pred_labels, average='weighted', pos_label=0)
auc_roc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)
print('Specificity:', specificity)
print('AUC-ROC:', auc_roc)

# Plot the accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Confusion Matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_labels)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_folders, rotation=45)
plt.yticks(tick_marks, class_folders)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
