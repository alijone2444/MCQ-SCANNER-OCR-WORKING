import cv2
import os
from rectanglesFinder import process_image_with_rectangles
from croppingRectangles import crop_all_rectangles
from perform_ocr import perform_ocr_on_cropped_image

# Path to the image file
image_path = 'IMG_20240522_142511.jpg'

rectangles_with_positions = process_image_with_rectangles(image_path)

rectangle_images = [rect[1] for rect in rectangles_with_positions]
rectangle_positions = [rect[0] for rect in rectangles_with_positions]

cropped_images, rectangle_positions = crop_all_rectangles(rectangle_images, rectangle_positions)

outputAnswers = {}

for idx, cropped_img in enumerate(cropped_images):
    position = rectangle_positions[idx]
    outputAnswers[position] = cropped_img

ocr_results = {}

for position, cropped_img in outputAnswers.items():
    position,text = perform_ocr_on_cropped_image(position, cropped_img)  # Assuming perform_ocr_on_cropped_image function is defined
    ocr_results[position] = text

print('answers', ocr_results)

sorted_contours = sorted(rectangle_positions, key=lambda x: (x[1], x[0]))
print(sorted_contours)
finalAnswer=[]
for i in sorted_contours:
    finalAnswer.append(ocr_results[i])
print(finalAnswer)
cv2.waitKey(0)
cv2.destroyAllWindows()
