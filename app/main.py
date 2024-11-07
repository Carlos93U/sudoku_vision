import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from src.preprocessing import find_sudoku_square, save_solution_image
from src.digit_recognition import load_trained_model, extract_digits
from src.solver import solve_sudoku

# Set up the page
st.title("Sudoku Vision")

# Create a widget to upload an image
st.markdown("## Upload a Sudoku image or take a photo")
img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

# Load the digit recognition model
@st.cache_resource()
def load_digit_model():
    return load_trained_model("./models/sudoku_digit_recognizer.h5")

model = load_digit_model()

# Function to create a download link for the image
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

if img_file_buffer is not None:
    # Read the uploaded image
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    # Process the image to obtain the aligned Sudoku
    warped_img = find_sudoku_square(image)

    # Create columns to display images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(warped_img, channels="BGR", caption="Aligned Sudoku")

    # Recognize digits and extract the Sudoku matrix
    sudoku_matrix, empty_cells = extract_digits(warped_img, model)

    # Solve the Sudoku
    if solve_sudoku(sudoku_matrix):
        # Generate and display the solution image
        save_solution_image(warped_img, sudoku_matrix, empty_cells, output_path="solution.png")
        solution_image = Image.open("solution.png")

        with col2:
            st.image(solution_image, caption="Solved Sudoku")

        # Link to download the solution image
        st.markdown(get_image_download_link(solution_image, "solved_sudoku.jpg", 'Download solution image'), unsafe_allow_html=True)
    else:
        st.write("Unable to solve the Sudoku puzzle. Please ensure the photo is clear and well-aligned, and try again.")
