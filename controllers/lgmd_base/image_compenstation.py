import cv2
import numpy as np

# ---- Load images ----
img1 = cv2.imread("Move/1.jpg")
img2 = cv2.imread("Move/3.jpg")

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ---- Feature detection (ORB) ----
orb = cv2.ORB_create(2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# ---- Match features ----
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by quality
matches = sorted(matches, key=lambda x: x.distance)

# Use the best N matches
good_matches = matches[:500]

# Extract matched keypoints
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# ---- Estimate homography (RANSAC handles outliers) ----
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# ---- Warp image2 to align with image1 ----
aligned2 = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

# Convert aligned image to grayscale
aligned2_gray = cv2.cvtColor(aligned2, cv2.COLOR_BGR2GRAY)

# ---- Subtract images ----
diff = cv2.absdiff(gray1, aligned2_gray)

# ---- Threshold for motion mask ----
_, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Optional: clean with morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

# ---- Save outputs ----
cv2.imwrite("aligned_frame.png", aligned2)
cv2.imwrite("difference.png", diff)
cv2.imwrite("motion_mask.png", motion_mask)

print("Done! Check aligned_frame.png and motion_mask.png")
