import cv2
import numpy as np

def make_panorama_scan(map, panorama_width, sr2br, out_size: tuple=None):
    # Define the circles
    center = (map.shape[1] // 2, map.shape[0] // 2)
    radius1 = map.shape[0] // 2
    radius2 = int(sr2br * radius1)
    half_width = int(panorama_width/2)
    diff = radius1-radius2
    # Create a black image for the panorama
    panorama1 = np.zeros((diff, half_width), dtype=np.uint8)
    panorama2 = np.zeros((diff, half_width), dtype=np.uint8)
    # Loop through each angle
    for a in range(1, half_width, 3):
        angle = (a-1)/(panorama_width/360)
        # Rotate the probability map
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(map, M, (map.shape[1], map.shape[0]))
        # Extract the portion of the rotated image that matches the circles
        mask1 = cv2.circle(np.zeros_like(rotated), center, radius1, 255, -1)
        mask2 = cv2.circle(np.zeros_like(rotated), center, radius2, 255, -1)
        masked_rotated = cv2.bitwise_and(rotated, rotated, mask=cv2.bitwise_xor(mask1, mask2))
        # Paste the column into the panorama
        panorama1[:, a-1:a+2] = masked_rotated[0:diff, center[0]-1:center[0]+2]
        panorama2[:, a-1:a+2] = np.flip(masked_rotated[-diff:, center[0]-1:center[0]+2], 0)
    panorama = cv2.hconcat([panorama1, panorama2])
    if out_size is None:
        return panorama
    else:
        return cv2.resize(panorama, out_size, interpolation=cv2.INTER_LINEAR)

if __name__ == '__main__':
    prob_map = cv2.imread("i.jpg", cv2.IMREAD_GRAYSCALE)
    panorama_width = 1080
    sr2br = 0.5
    panorama = make_panorama_scan(prob_map, panorama_width, sr2br, (1200, 400))
    cv2.imwrite("panorama.jpg", panorama)
