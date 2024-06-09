import cv2
import numpy as np

def crop_and_resize_paper(image_path):
    # Load the image
    orig = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Dilate to enhance edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, kernel)

    # Detect edges using Canny
    edges = cv2.Canny(dilated, 0, 84, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)

    # Simplify contours down to polygons
    rects = []
    for cont in contours:
        rect = cv2.approxPolyDP(cont, 40, True).reshape(-1, 2)
        rects.append(rect)

    # Find bounding boxes and crop the paper
    for rect in rects:
        x, y, w, h = cv2.boundingRect(rect)
        paper = orig[y:y + h, x:x + w]

        # Resize the paper image
        resized_paper = cv2.resize(paper, (400, 400))

        return resized_paper

