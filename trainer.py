import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model

# Define a ResNet-18-like architecture
def create_resnet18_model(input_shape=(32, 32, 1), num_classes=10):
    # Load pre-trained ResNet-50 model
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,  # Exclude the original classification head
        weights=None,       # Use random initialization (training from scratch)
        input_tensor=Input(shape=input_shape),
        input_shape=input_shape,
    )

    # Add custom layers for MNIST
    x = Flatten()(base_model.output)                # Flatten the output of the convolutional layers
    x = Dense(128, activation='relu')(x)            # Add a dense layer
    output = Dense(num_classes, activation='softmax')(x)  # Final layer for 10-class classification

    # Build the model
    model = Model(inputs=base_model.input, outputs=output)

    return model

# Create the model
model = create_resnet18_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = tf.image.resize(x_train[..., tf.newaxis], (32, 32)).numpy() / 255.0  # Resize to 32x32 and normalize
# x_train = np.repeat(x_train, 3, axis=-1)  # Convert grayscale (1 channel) to RGB (3 channels)

x_test = tf.image.resize(x_test[..., tf.newaxis], (32, 32)).numpy() / 255.0
# x_test = np.repeat(x_test, 3, axis=-1)  # Convert grayscale to RGB

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Save the trained model
model.save("resnet18_mnist_model.keras")

# Load the model
# model = tf.keras.models.load_model("mnist_model.keras")

# Test a few samples from the test set
for i in range(5):
    sample_image = x_test[i]  # Take one sample
    plt.imshow(sample_image, cmap="gray")  # Show the image
    plt.pause(2)

    # Reshape and predict
    prediction = np.argmax(model.predict(sample_image.reshape(1, 32, 32, 1)))
    print(f"Predicted Digit: {prediction}")
    
