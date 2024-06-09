import cv2

def crop_all_rectangles(rectangle_images,rectangle_positions):
    cropped_images = []
    # Define the coordinates (adjust as needed)
    left, upper, right, lower = 200, 30, 400, 200
    for idx, img in enumerate(rectangle_images):
        cropped_img = img[upper:lower, left:right]
        cropped_images.append(cropped_img)
       
    return cropped_images,rectangle_positions
