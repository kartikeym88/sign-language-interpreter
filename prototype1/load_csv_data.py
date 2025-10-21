import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Load CSVs
train_df = pd.read_csv('dataset/sign-language-mnist/sign_mnist_train.csv')
test_df = pd.read_csv('dataset/sign-language-mnist/sign_mnist_test.csv')

# Separate features and labels
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Remove labels 9 (J) and 25 (Z)
def filter_and_remap(X, y):
    valid_mask = (y != 9) & (y != 25)
    X = X[valid_mask]
    y = y[valid_mask]

    # Remap: compress the label space
    new_y = []
    for label in y:
        if label < 9:
            new_y.append(label)
        elif label < 25:
            new_y.append(label - 1)  # Shift down 1 for labels >9
    return X, np.array(new_y)

X_train, y_train = filter_and_remap(X_train, y_train)
X_test, y_test = filter_and_remap(X_test, y_test)

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=24)
y_test = to_categorical(y_test, num_classes=24)

# Show a sample image
plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.title("Sample Image")
plt.axis("off")
plt.show()
