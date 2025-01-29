# Digit Decoder
This project is a digit recognition application that uses a trained ResNet-18 model to predict handwritten digits. Users can draw a digit on a Pygame canvas, and the program will use the trained model to recognize and predict the digit. The predicted digit is printed in the terminal.

---

## Project Overview

This project allows users to:
1. Train a ResNet-18 model on the MNIST dataset for digit recognition.
2. Interact with the trained model by drawing digits on a Pygame canvas.
3. Obtain predictions from the model, which will be displayed in the terminal.

---

## Files in the Repository

1. **`trainer.py`**
   - Trains a ResNet-18 model on the MNIST dataset.
   - Saves the trained model as `resnet18_mnist_model.keras`.
   - Handles model evaluation on the test dataset, providing test accuracy and loss metrics.

2. **`canvas.py`**
   - A Pygame-based interactive canvas where users can draw digits using their mouse/trackpad.
   - Uses the trained ResNet-18 model (`resnet18_mnist_model.keras`) to predict digits.
   - The predicted digit is printed directly to the terminal when the user presses the spacebar.
   - Clear the canvas by clicking the `C` key.
     
3. **`resnet18_mnist_model.keras`**
   - Pre-trained ResNet-18 model file for digit recognition using the following parameters:
        - Epochs: 10
        - Batch Size: 64
        - Learning Rate: 0.001 (default for Adam optimizer)
        - Loss Function: Sparse Categorical Crossentropy
        - Validation Data: 10,000 test samples from the MNIST dataset
   - Used by `canvas.py` to make predictions without retraining the model.
---

## Future Plans
- [Short Term] Revamping Pygame Canvas to be more visually appealing
- [Short Term] Train model using Google Colab so that more epochs can be run during training
- [Short Term] Plot epoch times and accuracy using matplotlib
- [Long Term] Support for multi-digit numbers (i.e. recognizing '5' and '1' as '51')
