import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
 
def sift():
    left = cv.imread('left.jpg')
    right = cv.imread('right.jpg')

    grayleft = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    grayright = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(grayleft, None)
    keypoints2, descriptors2 = sift.detectAndCompute(grayright, None)

    kp1_img = cv.drawKeypoints(grayleft, keypoints1, left)
    kp2_img = cv.drawKeypoints(grayright, keypoints1, right)

    cv.imwrite('kp1_img.jpg', kp1_img)
    cv.imwrite('kp2_img.jpg', kp2_img)

    # feature matching

    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x : x.distance)

    result = cv.drawMatches(left, keypoints1, right, keypoints2, matches[:50], right, flags=2)
    cv.imwrite('result.jpg', result)
    plt.imshow(result)
    plt.show()


if __name__ == '__main__':
    sift()


