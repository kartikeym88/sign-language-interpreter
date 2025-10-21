import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Step 1: Load CSVs
train_df = pd.read_csv('sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test.csv')

# Step 2: Inspect the data (optional, just to check)
print(train_df.head())
print(test_df.head())

# Step 3: Separate features and labels
X_train = train_df.iloc[:, 1:].values  # All columns except the first one (pixels)
y_train = train_df.iloc[:, 0].values   # The first column is the labels

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Step 4: Remove labels 9 (J) and 25 (Z)
def filter_and_remap(X, y):
    valid_mask = (y != 9) & (y != 25)  # Exclude 'J' and 'Z'
    X = X[valid_mask]
    y = y[valid_mask]

    # Remap the labels to compress the space
    new_y = []
    for label in y:
        if label < 9:
            new_y.append(label)
        elif label < 25:
            new_y.append(label - 1)  # Shift down 1 for labels > 9
    return X, np.array(new_y)

# Apply the filter
X_train, y_train = filter_and_remap(X_train, y_train)
X_test, y_test = filter_and_remap(X_test, y_test)

# Step 5: Normalize pixel values
X_train = X_train / 255.0  # Normalize the pixel values to [0, 1]
X_test = X_test / 255.0

# Step 6: Reshape data to 28x28 images with 1 channel (grayscale)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Step 7: One-hot encode the labels
y_train = to_categorical(y_train, num_classes=24)  # One-hot encoding for training labels
y_test = to_categorical(y_test, num_classes=24)    # One-hot encoding for testing labels

# Step 8: Show a sample image
plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.title("Sample Image")
plt.axis("off")
plt.show()

# If you want to save the processed data, you can do so with:
# np.save('X_train.npy', X_train)
# np.save('y_train.npy', y_train)
# np.save('X_test.npy', X_test)
# np.save('y_test.npy', y_test)
