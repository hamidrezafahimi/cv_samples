import cv2
import os

# Path to your input image
input_image_path = "/media/hamid/Workspace/DATA/bagimgs/img0009.jpg"  # Update with your actual image path

# Read the input image
img = cv2.imread(input_image_path)

# Resize the image to 640x480
target_width, target_height = 640, 480
resized_img = cv2.resize(img, (target_width, target_height))

# Save the resized image in the home directory
output_image_path = os.path.expanduser('~/resized_image.jpg')
cv2.imwrite(output_image_path, resized_img)

print(f"Resized image saved at: {output_image_path}")
