import cv2
import numpy as np
from ransac import *
from knn import KnnClassifier


class Image_Stitching():
    """Image Stitching classifier."""
    
    def __init__(self, crossCheck=True):
        """Initialization of parameters."""
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.smoothing_window_size = 800
        self.matches1to2 = []
        self.good_points = []
        self.doCrossCheck = crossCheck #can be changed

    def crossCheck(self, first_matches, second_matches):
        """ Function to perform Cross-Check validation to exclude possibly
        incorrect matches. Returns matches1to2 - a list of matches from image1
        to image 2, and good_points, which is a list of indexes of matching keypoints.
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
        """Function to perform fearure matching. Computes matches for images
        and outputs matching result on the picture. Returns Homography matrix
        for futher calculations (Ransac)."""
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
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, self.matches1to2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('usa_matching.jpg', img3)

        if len(self.good_points) > self.min_match:
            image1_kp = np.float32([kp1[pair[0]].pt for pair in self.good_points])
            image2_kp = np.float32([kp2[pair[1]].pt for pair in self.good_points])

            # apply ransac and get the best homography matrix.
            H = ransac(image2_kp, image1_kp, 1000, 5.0, 0.8) #5.0 a threshold value (can be changed)
        return H

    def create_mask(self, img1, img2, version):
        height_img1, width_img1, width_img2 = img1.shape[0], img1.shape[1], img2.shape[1]
        height_panorama, width_panorama = height_img1, width_img1 + width_img2

        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))

        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
    
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        H = self.feature_matching(img1,img2)
        width = img1.shape[1] + img2.shape[1]
        height = max(img1.shape[0], img2.shape[0])
        result = cv2.warpPerspective(img2, H,  (width, height))
        result[0:img1.shape[0], 0:img1.shape[1]] = img1
        return result

def main(argv1,argv2):
    """ Main function to start stitching images."""
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    final=Image_Stitching().blending(img1,img2)
    cv2.imwrite('hello_america.jpg', final)

if __name__ == '__main__':
    try: 
        main('usa1.jpg', 'usa2.jpg')
    except IndexError:
        print ('Error occured. Input images.')
    