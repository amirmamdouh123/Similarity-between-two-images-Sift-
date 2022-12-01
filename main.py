import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from Utilities import *
images = os.listdir('assignment data')
path= 'assignment data/'
for im1 in images :
    img_path1=os.path.join(path,im1)
    image1=cv2.imread(img_path1,0)
    diffrance = []
    for im2 in images:
        if(im1==im2):
            continue;
        elif(im1[5]==im2[5]):
            img_path2 = os.path.join(path, im2)
            image2 = cv2.imread(img_path2, 0)
            #sift and get matches
            allMatches,kps1,desc1,kps2,desc2=sift_matches(image1,image2)

            # first way for filtering
            match,ratio=get_inliers_ratio(desc1,desc2,ratio=0.7)
            #or
            #second way for filtering
            #match,ratio= get_inliers_crossCheck(desc1,desc2)

            #calculate similaritiy
            similarity=getSimilarity(match,kps1,kps2)
            bool =if_similar(similarity,ratio)

            # Draw matches after compressing matches
            Draw_matches(image1, image2, kps1, kps2, match,similarity,bool)

            #in Case Work with homography
            # (H, status)= find_homography(match,kps1,kps2,reprojThresh=4.0)
            # Draw_matches_RANSAC(image1,image2,kps1,kps2,match,status,similarity,bool)


    # min_value = min(diffrance)
    # min_index = diffrance.index(min_value)
    # nearImage = os.path.join(path, images[min_index])
    # plt.imshow(cv2.imread(nearImage,0)),plt.text(-25, -10, 'nearest image', fontsize=15), plt.show()


















