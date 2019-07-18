import cv2
import numpy as np

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


def meth(feature_detector, img, frame, grayframe, flann, number):
    if feature_detector == "SIFT":
        print("SIFT")
        # SIFT Features Detector
        sift = cv2.xfeatures2d.SIFT_create()

        # features of picture
        kp_image, desc_image = sift.detectAndCompute(img, None)

        # features of grayframe
        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)


    elif feature_detector == "SURF":
        print("SURF")
        # SURF Features Detector
        surf = cv2.xfeatures2d.SURF_create()

        # features of picture
        kp_image, desc_image = surf.detectAndCompute(img, None)

        # features of grayframe
        kp_grayframe, desc_grayframe = surf.detectAndCompute(grayframe, None)

    # find matches using flann (FlannBasedMatcher - объект для матчинга)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    # отбираем только хорошие совпадения
    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    # matches = sorted(matches, key=lambda x: x.distance)
    good_points = sorted(good_points, key=lambda x: x.distance)

    # создаём картинку, отображающую совпадения
    img_with_matching = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points[:15], grayframe)

    # Homography(Гомография) - перспективная трансформация
    # если число совпадений> 10, ищем гомографию, иначе - отображаем просто grayframe
    if (len(good_points) > 10):

        good_points_truncated = good_points[:15]

        # get coordinates of picture(query) and grayframe(train) keypoints
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points_truncated]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points_truncated]).reshape(-1, 1, 2)

        # находим гомографию, матрицу перспективной трансформации между двумя картинками(query и train)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Perspective transform
        h, w = img.shape  # размеры картинки(query)

        # создаём рамку для обозначения совпадающей картинки
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        # define color of marking
        if (number == 1):
            color = blue
        elif (number == 2):
            color = green
        elif (number == 3):
            color = red

        # делаем перспективную трансформацию рамки параллельно картинке
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, color, 3)

        # compute the center of the contour
        M = cv2.moments(dst)
        if M["m00"] == 0.0:
            cX = int(M["m10"] / (M["m00"] + 0.00000001))
            cY = int(M["m01"] / (M["m00"] + 0.00000001))
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        # draw the contour and center of the shape on the image
        # cv2.drawContours(homography, [dst], -1, (0, 255, 0), 2)
        cv2.circle(homography, (cX, cY), 20, color, 3)
        cv2.putText(homography, str(number), (cX - 5, cY + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("Homography", grayframe)

    return img_with_matching


def main():
    feature_detector = "SIFT"
    # feature_detector = "SURF"

    path_to_video = '../resources/beer_table.avi'

    path_to_template1 = '../resources/beer1.png'
    img1 = cv2.imread(path_to_template1, cv2.IMREAD_GRAYSCALE)

    path_to_template2 = '../resources/beer2.png'
    img2 = cv2.imread(path_to_template2, cv2.IMREAD_GRAYSCALE)

    path_to_template3 = '../resources/beer3.png'
    img3 = cv2.imread(path_to_template3, cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture(path_to_video)

    # FlannBasedMatcher - объект для матчинга с параметрами.
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    while True:

        ret, frame = cap.read()

        if ret == True:
            # convert to gray
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            img_with_matching1 = meth(feature_detector, img1, frame, grayframe, flann, 1)
            img_with_matching2 = meth(feature_detector, img2, frame, grayframe, flann, 2)
            img_with_matching3 = meth(feature_detector, img3, frame, grayframe, flann, 3)

            # cv2.imshow("Image1", img1)
            # cv2.imshow("grayFrame", grayframe)
            cv2.imshow("img_with_matching1", img_with_matching1)
            cv2.imshow("img_with_matching2", img_with_matching2)
            cv2.imshow("img_with_matching3", img_with_matching3)

            # if 'Esc' (k==27) is pressed then break
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
