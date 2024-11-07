import cv2
import numpy as np
import os
import random
from datetime import datetime

# Directories for original cells and augmented images
original_dir = "data/cells"
augmented_dir = "data/augmented_cells"

# Create folders for each digit in 'data/augmented_cells' (1 to 9)
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

for digit in range(1, 10):
    digit_dir = os.path.join(augmented_dir, str(digit))
    if not os.path.exists(digit_dir):
        os.makedirs(digit_dir)

# Function to apply data augmentation without noise
def augment_image(img):
    # Random rotation between -3 and 3 degrees
    angle = random.uniform(-3, 3)
    M_rot = cv2.getRotationMatrix2D((14, 14), angle, 1)  # Center at the midpoint of 28x28
    rotated = cv2.warpAffine(img, M_rot, (28, 28))

    # Random translation between -2 and 2 pixels
    tx, ty = random.randint(-2, 2), random.randint(-2, 2)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(rotated, M_trans, (28, 28))

    # Slight scaling
    scale_factor = random.uniform(0.9, 1.0)
    scaled = cv2.resize(translated, (0, 0), fx=scale_factor, fy=scale_factor)
    if scaled.shape[0] > 28 or scaled.shape[1] > 28:
        scaled = cv2.resize(scaled, (28, 28))  # Resize if it exceeds the limit

    # Adjust brightness and contrast
    brightness = random.randint(-30, 30)
    contrast = random.uniform(0.9, 1.1)
    adjusted = cv2.convertScaleAbs(scaled, alpha=contrast, beta=brightness)

    # Final resizing to 28x28 if necessary
    final_augmented = cv2.resize(adjusted, (28, 28))
    return final_augmented

# Apply augmentation to each cell in subfolders 'data/cells/[digit]' and save to 'data/augmented_cells/[digit]'
for digit in range(1, 10):  # Only digits 1 to 9
    digit_path = os.path.join(original_dir, str(digit))
    augmented_digit_path = os.path.join(augmented_dir, str(digit))
    
    for filename in os.listdir(digit_path):
        img_path = os.path.join(digit_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Generate multiple augmented images per original image
        for i in range(4):  # Generate 4 augmentations per image
            augmented_img = augment_image(img)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")  # Timestamp for unique filenames
            cv2.imwrite(os.path.join(augmented_digit_path, f"{timestamp}.png"), augmented_img)
