import pygame
import numpy as np
import tensorflow as tf

# Initialize Pygame
pygame.init()

# Screen Dimensions
WIDTH, HEIGHT = 300, 300
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Digit Recognition Canvas")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Canvas setup
screen.fill(BLACK)
drawing = False  # To check if the user is drawing
radius = 10  # Brush size

# Load Pre-trained Model
model = tf.keras.models.load_model("resnet18_mnist_model.keras")

# Model evaluation (prints loss and accuracy)
(x_test, y_test) = tf.keras.datasets.mnist.load_data()[1]
x_test = tf.image.resize(x_test[..., np.newaxis], (32, 32)).numpy() / 255.0  # Resize and normalize
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Model Evaluation:\n  Test Loss: {loss:.5f}\n  Test Accuracy: {accuracy:.5f}")
print("READY! You can start drawing.")

# Helper function to preprocess the drawing
def preprocess(surface):
    """Convert the Pygame canvas to a 32x32 grayscale image for the model."""
    # Extract raw pixel array from the surface
    raw_array = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # Convert to (H, W, C)

    # Convert to grayscale
    gray_array = np.dot(raw_array[..., :3], [0.2989, 0.5870, 0.1140])  # Weighted sum for grayscale

    # Create a surface from the grayscale array
    gray_surface = pygame.surfarray.make_surface(gray_array).convert_alpha()  # Ensure proper pixel format

    # Resize to 32x32 using smoothscale
    resized_surface = pygame.transform.smoothscale(gray_surface, (32, 32))

    # Convert resized surface back to an array
    resized_array = pygame.surfarray.array3d(resized_surface).mean(axis=-1)

    # Normalize pixel values (0 to 1) and reshape for the model
    return resized_array / 255.0

# Main Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True  # Start drawing
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False  # Stop drawing
        elif event.type == pygame.MOUSEMOTION and drawing:
            # Draw as the mouse moves
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, WHITE, (mouse_x, mouse_y), radius)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # Spacebar to predict
                try:
                    input_image = preprocess(screen).reshape(1, 32, 32, 1)
                    prediction = np.argmax(model.predict(input_image))
                    print(f"Predicted Digit: {prediction}")
                    screen.fill(BLACK)  # Clear the screen after prediction
                except Exception as e:
                    print(f"Error during prediction: {e}")
            elif event.key == pygame.K_c:  # Clear the screen
                screen.fill(BLACK)

    pygame.display.flip()

pygame.quit()
