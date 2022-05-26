import numpy as np


class KnnClassifier():
    """Knn classifier for feature matching."""

    def __init__(self):
        """ Initialization of KNN classifier with data."""
        self.match_dictionary = {}
        self.ratio = 0.8
    
    def classify(self, ind, point, descriptors2, k = 2):
        """Return a match to a point by finding nearest neighbors and
        performing Lowe's ratio test to find the best single match from
        keypoints of image 2 for initial keypoint from image 1."""
        # compute distance to all training points.
        matches = [(np.linalg.norm(point - descriptors2[i]), i) for i in range(len(descriptors2))]

        # sort them by distance to get nearest neighbors.
        matches.sort(key = lambda tup: tup[0])
        two_neighbors = matches[:k]
        
        #Lowe's ratio test.
        if two_neighbors[0][0] < two_neighbors[1][0] * self.ratio:
            # use dictionary to store index of nearest matching keypoint for initial point.
            self.match_dictionary[ind] = two_neighbors[0][1]
        return self.match_dictionary
    
    def knnMatch(self, descriptors1, descriptors2, k):
        """Function to perform K-Nearest-Neighbors algorithm for each descriptor
        from image 1 to find nearest neighbors from image 2."""
        for i in range(len(descriptors1)):
            self.classify(i, descriptors1[i], descriptors2, k)
        return self.match_dictionary
