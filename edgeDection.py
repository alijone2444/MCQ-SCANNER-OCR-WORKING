import cv2
import os
import pytesseract
import numpy as np

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def resize_image_to_fit_screen(image, max_height=900):
    aspect_ratio = image.shape[1] / image.shape[0]
    width = int(max_height * aspect_ratio)
    resized_image = cv2.resize(image, (width, max_height), interpolation=cv2.INTER_AREA)
    return resized_image

def show_image(window_name, image, max_height=900):
    resized_image = resize_image_to_fit_screen(image, max_height)
    cv2.imshow(window_name, resized_image)

# Path to the image file
script_dir = os.path.dirname(__file__)
image_path = os.path.join(script_dir, 'IMG_20240522_142511.jpg')

# Read the image
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image.")
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Canny edge detection
canny_edges = cv2.Canny(blurred_image, 100, 200)
# Apply Laplacian filter
laplacian_image = cv2.Laplacian(canny_edges, cv2.CV_64F)
laplacian_image = cv2.convertScaleAbs(laplacian_image)

# Apply thresholding
_, thresholded_image = cv2.threshold(laplacian_image, 128, 255, cv2.THRESH_BINARY)

blurred_image = cv2.GaussianBlur(thresholded_image, (5, 5), 0)

# Show the thresholded image
show_image('Thresholded Image', blurred_image)

# Find contours
contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out rectangles
rectangles = []
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        rectangles.append(approx)

# Define a more aggressive sharpening kernel
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1,10,-1],
                              [-1,-1,-1]])

# Create a copy of the original image for the overlay
image_with_highlight = image.copy()

# Process each rectangle separately
for idx, rectangle in enumerate(rectangles):
    x, y, w, h = cv2.boundingRect(rectangle)

    # Extract the region of interest
    roi = gray_image[y:y+h, x:x+w]

    # Highlight the region of interest with a semi-transparent overlay
    overlay = image_with_highlight.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)  # Red overlay
    alpha = 0.4  # Transparency factor
    image_with_highlight = cv2.addWeighted(overlay, alpha, image_with_highlight, 1 - alpha, 0)

    # Sharpen the region of interest
    sharpened_roi = cv2.filter2D(roi, -1, kernel_sharpening)

    # Use pytesseract to recognize text in the sharpened region of interest
    text = pytesseract.image_to_string(sharpened_roi, config='--psm 7')
    print(f"Text in Rectangle {idx + 1}: {text.strip()}")

    # Optionally, draw the recognized text on the original image with highlight
    cv2.putText(image_with_highlight, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Create a new window for each ROI and display it
    roi_window_name = f'Rectangle {idx + 1}'
    cv2.imshow(roi_window_name, sharpened_roi)

# Resize and show the final image with rectangles and text
show_image('Original Image with Highlighted Rectangles and Text', image_with_highlight)

# Wait for a key press and close all image windows
cv2.waitKey(0)
cv2.destroyAllWindows()
