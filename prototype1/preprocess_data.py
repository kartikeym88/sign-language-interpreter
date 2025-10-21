import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the directory where the images are located
train_dir = 'C:/Users/KARTIKEY MADAAN/OneDrive/Desktop/SignLanguageInterpreter/dataset/sign-language-mnist'

# Preprocessing the images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load and iterate training data
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Resize images to 64x64 pixels
    batch_size=32,
    class_mode='categorical',  # Use categorical labels (for classification)
    subset='training'  # Set aside 20% for validation
)

# Load and iterate validation data
valid_data = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',  # Use categorical labels
    subset='validation'  # This will be the validation data
)
