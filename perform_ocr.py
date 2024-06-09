import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def perform_ocr_on_cropped_image(position, cropped_image):
    try:
        # Apply Laplacian filter
        laplacian = cv2.Laplacian(cropped_image, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(laplacian, (3, 3), 0)
        
        # Perform OCR on the filtered image
        text = pytesseract.image_to_string(blurred, config='--psm 7')
        print(text)
        
        # Display the filtered image
        cv2.imshow(f'Laplacian and Gaussian Blurred Image at Position {position}', blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return position, text
    except Exception as e:
        print("An error occurred during OCR processing:", e)
        return position, None
