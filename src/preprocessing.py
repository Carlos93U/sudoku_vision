import cv2
import numpy as np
import operator

def find_sudoku_square(image):
    """
    Preprocess the input image to identify and extract the main Sudoku square.
    
    Parameters:
    - image (numpy.ndarray): The input image containing the Sudoku puzzle.
    
    Returns:
    - grid_image (numpy.ndarray): The warped image with grid lines, focused on the main Sudoku square.
    """

    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur_img = cv2.GaussianBlur(gray_img, (9, 9), 0)

    # Binarize the image using adaptive thresholding
    binary_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert the image colors for contour detection
    negative_img = cv2.bitwise_not(binary_img)

    # Apply dilation to strengthen contour lines
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dilated_img = cv2.dilate(negative_img, kernel, iterations=1)

    # Find the contours and select the largest one (assumed to be the Sudoku grid)
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # Identify corner points
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in max_contour]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in max_contour]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in max_contour]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in max_contour]), key=operator.itemgetter(1))

    # Extract the corner points
    bottom_right = max_contour[bottom_right][0]
    top_left = max_contour[top_left][0]
    bottom_left = max_contour[bottom_left][0]
    top_right = max_contour[top_right][0]
    
    # Prepare for perspective transformation by defining source and destination points
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([
        np.linalg.norm(bottom_right - top_right),
        np.linalg.norm(top_left - bottom_left),
        np.linalg.norm(bottom_right - bottom_left),
        np.linalg.norm(top_left - top_right)
    ])

    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype='float32')

    # Apply the perspective transformation to obtain a straightened view of the Sudoku grid
    m = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(image, m, (int(side), int(side)))
    
    # Draw grid lines
    grid_image = draw_grid_lines(warped_img)

    return grid_image


def draw_grid_lines(warped_img):
    """
    Draws grid lines on the warped Sudoku image to delineate the boundaries between cells.
    
    Parameters:
    - warped_img (numpy.ndarray): Warped image of the main Sudoku square.
    
    Returns:
    - grid_image (numpy.ndarray): Image with grid lines drawn for cell boundaries.
    """
    
    # Get dimensions of the image
    height, width = warped_img.shape[:2]
    cell_height = height // 9  # Cell height for 9x9 grid
    cell_width = width // 9    # Cell width for 9x9 grid

    # Create a copy of the image to draw the grid lines
    grid_image = warped_img.copy()

    # Draw horizontal lines, including the top and bottom edges
    for i in range(10):  # Draw 10 lines to cover all cells
        y = min(i * cell_height, height - 1)  # Ensure lines are within bounds
        cv2.line(grid_image, (0, y), (width - 1, y), (0, 0, 0), 5)  # Black line with thickness 5

    # Draw vertical lines, including the left and right edges
    for i in range(10):  # Draw 10 lines to cover all cells
        x = min(i * cell_width, width - 1)  # Ensure lines are within bounds
        cv2.line(grid_image, (x, 0), (x, height - 1), (0, 0, 0), 5)  # Black line with thickness 5

    return grid_image

def save_solution_image(grid_image, sudoku_matrix, empty_cells, output_path="solution.png"):
    """
    Draws the solved Sudoku numbers in the empty cells and saves the resulting image.
    
    Parameters:
    - grid_image (numpy.ndarray): Grid image of the Sudoku.
    - sudoku_matrix (numpy.ndarray): 9x9 matrix containing the solved Sudoku numbers.
    - empty_cells (numpy.ndarray): 9x9 matrix where 1 indicates an originally empty cell.
    - output_path (str): Path to save the image with the solution.
    """
    # Get dimensions of each cell
    height, width = grid_image.shape[:2]
    cell_height = height // 9
    cell_width = width // 9

    # Draw the solution in the empty cells
    for i in range(9):
        for j in range(9):
            if empty_cells[i, j] == 1:  # Only fill initially empty cells
                text = str(sudoku_matrix[i, j])
                x = j * cell_width + cell_width // 3
                y = i * cell_height + 2 * cell_height // 3
                cv2.putText(grid_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)  # Blue color for digits

    # Save the final image with the solution
    cv2.imwrite(output_path, grid_image)
    print(f"Solution saved as {output_path}")
