import cv2
import pytesseract

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def crop_image(image, left, upper, right, lower):
    cropped_img = image[upper:lower, left:right]
    return cropped_img

# Load the image using OpenCV
image_path = './rois/sharpened_roi_9.jpg'
img = cv2.imread(image_path)

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply the Laplacian filter
laplacian_img = cv2.Laplacian(gray_img, cv2.CV_64F)
laplacian_img = cv2.convertScaleAbs(laplacian_img)

# Crop the Laplacian image (example coordinates, adjust as needed)
left = 200
upper = 30
right = 400
lower = 200
cropped_img = crop_image(laplacian_img, left, upper, right, lower)

# Show the cropped Laplacian image
cv2.imshow('Cropped Laplacian Image', cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perform OCR
text = pytesseract.image_to_string(cropped_img, config='--psm 7')

# Print the extracted text
print("Extracted Text:")
print(text)
