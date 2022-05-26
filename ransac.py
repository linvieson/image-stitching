import numpy as np

def get_homography(image1,image2):
    #calculate homography matrix
    matrix = [[image1[0], image1[1], 1, 0, 0, 0,-image2[0]*image1[0],-image2[0]*image1[1]],
              [0, 0, 0, image1[0], image1[1], 1,-image2[1]*image1[0],-image2[1]*image1[1]],
              [image1[2], image1[3], 1, 0, 0, 0,-image2[2]*image1[2],-image2[2]*image1[3]],
              [0, 0, 0, image1[2], image1[3], 1,-image2[3]*image1[2],-image2[3]*image1[3]],
              [image1[4], image1[5], 1, 0, 0, 0,-image2[4]*image1[4],-image2[4]*image1[5]],
              [0, 0, 0, image1[4], image1[5], 1,-image2[5]*image1[4],-image2[5]*image1[5]],
              [image1[6], image1[7], 1, 0, 0, 0,-image2[6]*image1[6],-image2[6]*image1[7]],
              [0, 0, 0, image1[6], image1[7], 1,-image2[7]*image1[6],-image2[7]*image1[7]]]

    #apply homography on image2
    values = np.linalg.solve(matrix, image2)
    #get the result of mapping
    values = [[values[0],values[1],values[2]],[values[3],values[4],values[5]],[values[6],values[7],1]]
    return values


def ransac(img_1_points,img_2_points,iterations,threshold,ratio):
    #best model --> None
    model_out = np.array([])
    length_img1 = len(img_1_points)
    #A homography can be computed when there are 4 or more corresponding points
    # in two images.
    if length_img1 < 4:
        print("Not enough match points for RANSAC. Required minimum: 4")
        exit
    # counter for iterations  
    count = 0
    # best ratio (number of inliers / number of key points)
    best_ratio = 0
    while(count <= iterations):
        count += 1      
        #select 4 random keypoints from each image  
        random_indices = np.array(np.random.randint(0,length_img1-1,4))
        image1 = list(img_1_points[random_indices].flatten())
        image2 = list(img_2_points[random_indices].flatten())
        try:
            #calculate the homography
            matrix = np.array(get_homography(image1,image2))
        except:
            continue

        #counter for number of inliers
        inliers = 0
        #calculate the total error between mapped and actual points
        total_error = 0
        for j in range(length_img1):
            image1_point = img_1_points[j]
            image2_point = img_2_points[j]            
            transformed = matrix.dot(np.array([image1_point[0],image1_point[1],1]))            
            transformed = np.array([transformed[0]/transformed[2],transformed[1]/transformed[2]])
            # calculate the distance between actual point and transformed
            distance = np.linalg.norm(transformed-image2_point)
            
            # if distance is below a threshold value, it is considered an inlier
            if distance < threshold:
                inliers += 1
                total_error += distance
        
        #check whether current model is better than current best.
        if best_ratio < inliers/length_img1:
            best_ratio = inliers/length_img1
            model_out = matrix
            
    #if required ratio is satisfied --> return a model
    if best_ratio > ratio:        
        return model_out
    else:
        # lower the required ratio by 0.05 and find the best model.
        return ransac(img_1_points,img_2_points,iterations,threshold,ratio-0.05)
