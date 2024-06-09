import cv2
import os
import numpy as np

def resize_image_to_fit_screen(image, max_height=900):
    aspect_ratio = image.shape[1] / image.shape[0]
    width = int(max_height * aspect_ratio)
    resized_image = cv2.resize(image, (width, max_height), interpolation=cv2.INTER_AREA)
    return resized_image

def show_image(window_name, image, max_height=900):
    resized_image = resize_image_to_fit_screen(image, max_height)
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image_with_rectangles(img_path):
    # Read the image
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Unable to load image.")
        return None
    
    # Show the original image
    show_image("Original Image", image)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image("Grayscale Image", gray_image)

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    show_image("Blurred Image", blurred_image)

    # Apply Canny edge detection
    canny_edges = cv2.Canny(blurred_image, 100, 200)
    show_image("Canny Edges", canny_edges)

    # Apply Laplacian filter
    laplacian_image = cv2.Laplacian(canny_edges, cv2.CV_64F)
    laplacian_image = cv2.convertScaleAbs(laplacian_image)
    show_image("Laplacian Image", laplacian_image)

    # Apply thresholding
    _, thresholded_image = cv2.threshold(laplacian_image, 128, 255, cv2.THRESH_BINARY)
    show_image("Thresholded Image", thresholded_image)

    # Apply Gaussian Blur again
    blurred_image = cv2.GaussianBlur(thresholded_image, (5, 5), 0)
    show_image("Blurred Thresholded Image", blurred_image)

    # Find contours
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out rectangles
    rectangles = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            rectangles.append(approx)

    # Define a sharpening kernel
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 10, -1],
                                  [-1, -1, -1]])

    # Create a copy of the original image for the overlay
    image_with_highlight = image.copy()
    roi_sharpened = []

    # Process each rectangle separately
    rectangles_with_positions = []
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
        roi_sharpened.append(sharpened_roi)

        # Append the position of the rectangle along with the sharpened ROI image
        rectangles_with_positions.append(((x, y, w, h), sharpened_roi))

    # Show the final image with highlighted rectangles
    show_image("Final Image with Highlighted Rectangles", image_with_highlight)

    return rectangles_with_positions

