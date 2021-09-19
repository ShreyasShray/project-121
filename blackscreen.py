import cv2
import time
import numpy as np

# To save the output in a file output.avi .
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

# Taking background image
image = cv2.imread("image1.png")
image = cv2.resize(image, (640, 480))

# Starting the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# allowing the webcam to warmup for two seconds
time.sleep(2)

# Reading the capture frame until the camera is open
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    # flipping the image
    frame  = np.flip(frame, axis = 1)

    # Converting the bgr to hsv 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Generating mask to detect the red color
    u_black = np.array([104, 153, 70])
    l_black = np.array([30, 30, 0])

    mask_1 = cv2.inRange(frame, l_black, u_black)
    mask_2 = cv2.bitwise_not(mask_1)
    
    # Keeping only the part of images without the red color
    res1 = cv2.bitwise_and(frame, frame, mask = mask_2)

    # Keeping the part of image with the red color
    res2 = cv2.bitwise_and(image, image, mask = mask_1)

    # Generating the final output by merging both the res1 and res2
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    output_file.write(final_output)

    # Displaying the output to the user
    cv2.imshow("Magic", final_output)

    cv2.waitKey(1)

cap.release()
output_file.release()

cv2.destroyAllWindows()