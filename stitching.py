import cv2
from cv2 import ROTATE_90_CLOCKWISE
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from knn import KnnClassifier
from ransac import *

class Image_Stitching():
    """
    Image Stitching classifier.
    """

    def __init__(self, crossCheck=True):
        """
        Initialization of parameters.
        """
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.smoothing_window_size = 800
        self.matches1to2 = []
        self.good_points = []
        self.doCrossCheck = crossCheck #can be changed


    def crossCheck(self, first_matches, second_matches):
        """
        Function to perform Cross-Check validation to exclude possibly
        incorrect matches. Returns matches1to2 - a list of matches from image1
        to image 2, and good_points, which is a list of indexes of matching
        keypoints.
        """
        for key, value in first_matches.items():
            #if no keypoint1 for any keypoint2 to match we ignore it.
            if value not in second_matches:
                continue
            #if keypoint1 and keypoint2 match in both ways.
            elif second_matches[value] == key:
                self.matches1to2.append(cv2.DMatch(key, value, 1))
                self.good_points.append((key, value))
            #otherwise ignore the keypoint too.
            else:
                continue


    def feature_matching(self, img1, img2):
        """
        Function to perform feature matching. Computes matches for images
        and outputs matching result on the picture. Returns Homography matrix
        for futher calculations (Ransac).
        """
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        matcher = KnnClassifier()
        #find matches for image1 keypoints from image2 keypoints.
        first_matches = matcher.knnMatch(des1, des2, k=2)
        
        #check whether to perform CrossCheck validation.
        if self.doCrossCheck is True:
            matcher = KnnClassifier()
            #find matches for image2 keypoints from image1 keypoints.
            second_matches = matcher.knnMatch(des2, des1, k=2)
            #perform CrossCheck validation.
            self.crossCheck(first_matches, second_matches)

        elif self.doCrossCheck is False:
            for key, value in first_matches.items():
                self.matches1to2.append(cv2.DMatch(key, value, 1))
                self.good_points.append((key, value))
        
        # draw matches on picture.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, self.matches1to2, None, \
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('matching.jpg', img3)

        if len(self.good_points) > self.min_match:
            image1_kp = np.float32([kp1[pair[0]].pt for pair in self.good_points])
            image2_kp = np.float32([kp2[pair[1]].pt for pair in self.good_points])

            # apply ransac and get the best homography matrix.
            H = ransac(image2_kp, image1_kp, 1000, 5.0, 0.8)
            # 5.0 a threshold value (can be changed)
        else:
            raise(ValueError, 'Too few keypoints!')

        return H


    def stitch(self, img1, img2):
        H = self.feature_matching(img1, img2)
        width = img1.shape[1] + img2.shape[1]
        height = max(img1.shape[0], img2.shape[0])
        result = cv2.warpPerspective(img2, H,  (width, height))
        result[0:img1.shape[0], 0:img1.shape[1]] = img1

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]

        return final_result


def read_input():
    """
    Read the input paths to the images.
    """
    print('Enter path to images you want to stitch, one by one,\
 divided by enter.')
    images = []

    while True:
        inpt = input('Path to image: ')
        if inpt:
            images.append(inpt)
        else:
            break

    return images


def main(argv1, argv2, index, rotate=False):
    """
    Main function to start stitching images.
    """
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)

    if rotate:
        img1 = cv2.rotate(img1, ROTATE_90_CLOCKWISE)
        img2 = cv2.rotate(img2, ROTATE_90_CLOCKWISE)

    final = Image_Stitching().stitch(img1, img2)

    if rotate:
        final = cv2.rotate(final, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f'picture{index}.jpg', final)

    return f'picture{index}.jpg'


if __name__ == '__main__':
    rotation = input('Input H if you want to stitch images horizintally,\
 V if vertically: ')
    rotate = True if rotation == 'V' else False

    images = read_input()
    counter = 0
    rev_images = images[::-1]
    
    for ind, image in enumerate(rev_images[:-1]):
        new_path = main(images[ind], images[ind+1], counter, rotate)
        images[ind+1] = new_path
        counter += 1
    
    img = mpimg.imread(images[-1])
    imgplot = plt.imshow(img)
    plt.show()

