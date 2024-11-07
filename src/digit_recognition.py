import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_trained_model(model_path):
    """Load the trained digit recognition model."""
    return load_model(model_path)

def extract_digits(warped_img, model):
    """
    Extract digits from the warped Sudoku image using the model.
    Returns a matrix with detected digits and a mask for empty cells.
    """
    sudoku_matrix = np.zeros((9, 9), dtype=int)
    empty_cells = np.zeros((9, 9), dtype=int)
    height, width = warped_img.shape[:2]
    cell_height, cell_width = height // 9, width // 9
    gray_grid_image = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    margin = 20

    for i in range(9):
        for j in range(9):
            x_start, y_start = j * cell_width, i * cell_height
            cell = gray_grid_image[y_start:y_start + cell_height, x_start:x_start + cell_width]
            cell = clahe.apply(cell)
            cell = cv2.normalize(cell, None, 0, 255, cv2.NORM_MINMAX)
            _, binary_cell = cv2.threshold(cell, 90, 255, cv2.THRESH_BINARY_INV)
            roi = binary_cell[margin:cell_height-margin, margin:cell_width-margin]

            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and max(cv2.contourArea(c) for c in contours) >= 20:
                digit_img = preprocess_cell_for_model(contours, binary_cell, margin)
                predicted_digit, confidence = predict_digit(model, digit_img)
                if confidence >= 99:
                    sudoku_matrix[i, j] = predicted_digit
                else:
                    empty_cells[i, j] = 1
            else:
                empty_cells[i, j] = 1

    return sudoku_matrix, empty_cells

def preprocess_cell_for_model(contours, binary_cell, margin):
    """Extract and resize the cell's digit for prediction."""
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            x, y, w, h = cv2.boundingRect(contour)
            digit_region = binary_cell[y + margin:y + h + margin, x + margin:x + w + margin]
            aspect_ratio = w / h
            new_w, new_h = (28, int(28 / aspect_ratio)) if w > h else (int(28 * aspect_ratio), 28)
            final_cell = np.zeros((28, 28), dtype=np.uint8)
            x_offset, y_offset = (28 - new_w) // 2, (28 - new_h) // 2
            final_cell[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = cv2.resize(digit_region, (new_w, new_h))
            return final_cell

def predict_digit(model, digit_img):
    """Predict the digit in the cell using the model."""
    img_input = digit_img.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)
    prediction = model.predict(img_input)
    return np.argmax(prediction, axis=1)[0] + 1, np.max(prediction) * 100
