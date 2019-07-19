import cv2
import numpy as np

path_to_video = '../resources/beer_table.avi'

path_to_template1 = '../resources/beer1.png'
img1 = cv2.imread(path_to_template1, cv2.IMREAD_GRAYSCALE)

path_to_template2 = '../resources/beer2.png'
img2 = cv2.imread(path_to_template2, cv2.IMREAD_GRAYSCALE)

path_to_template3 = '../resources/beer3.png'
img3 = cv2.imread(path_to_template3, cv2.IMREAD_GRAYSCALE)

# Store width and height of templates
w1, h1 = img1.shape[::-1]
w2, h2 = img2.shape[::-1]
w3, h3 = img3.shape[::-1]


cap = cv2.VideoCapture(path_to_video)

while True:

    ret, frame = cap.read()

    if ret == True:
        # convert to gray
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform match operations.
        res1 = cv2.matchTemplate(grayframe, img1, cv2.TM_CCOEFF_NORMED)
        res2 = cv2.matchTemplate(grayframe, img2, cv2.TM_CCOEFF_NORMED)
        res3 = cv2.matchTemplate(grayframe, img3, cv2.TM_CCOEFF_NORMED)

        # Specify a threshold
        threshold = 0.8

        # Store the coordinates of matched area in a numpy array
        loc1 = np.where(res1 >= threshold)
        loc2 = np.where(res2 >= threshold)
        loc3 = np.where(res3 >= threshold)

        # Draw a rectangle around the matched region.
        for pt1 in zip(*loc1[::-1]):
            cv2.rectangle(frame, pt1, (pt1[0] + w1, pt1[1] + h1), (255, 0, 0), 2)

        for pt2 in zip(*loc2[::-1]):
            cv2.rectangle(frame, pt2, (pt2[0] + w2, pt2[1] + h2), (0, 255, 0), 2)

        for pt3 in zip(*loc3[::-1]):
            cv2.rectangle(frame, pt3, (pt3[0] + w3, pt3[1] + h3), (0, 0, 255), 2)

        # Show the final image with the matched area.
        cv2.imshow('Template Detection', frame)

        # if 'Esc' (k==27) is pressed then break
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
