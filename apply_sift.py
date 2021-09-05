import numpy as np
import cv2 as cv
import os
import pandas as pd
from numpy import linalg as LA

min_match_count = 10
# FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

sift = cv.SIFT_create()
paintings_path = './data/paintings/'
paintings = os.listdir(paintings_path)
paintings_results = {}

for painting in iter(paintings):
    painting_name = painting.split(".")[0]
    painting_img = cv.imread(os.path.join(paintings_path, painting), cv.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(painting_img, None)
    paintings_results.update({painting_name: [keypoints, descriptors]})

crops = []

for root, directories, files in os.walk('.\\objects\\'):
    crops.extend([os.path.join(root, name) for name in files])

metrics = []

for crop_path in iter(crops):
    print(crop_path)
    crop_img = cv.imread(crop_path, cv.IMREAD_GRAYSCALE)
    crop_kpts, crop_dpts = sift.detectAndCompute(crop_img, None)

    if crop_dpts is None:
        no_of_descriptors = 0
    else:
        no_of_descriptors = len(crop_dpts)

    if no_of_descriptors <= 2:

        metrics.append([crop_path, "-", no_of_descriptors] + [np.nan for i in range(8)])
    else:
        for painting_name, results in paintings_results.items():
            painting_kpts = results[0]
            painting_dpts = results[1]

            matches = flann.knnMatch(crop_dpts, painting_dpts, k=2)
            good_matches = []
            for m, n in matches:
                # test Lowe'a
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
            distances = [m.distance for m in good_matches]
            no_of_matches = len(good_matches)
            if no_of_matches < min_match_count:
                metrics.append(
                    [crop_path, painting_name, no_of_descriptors, no_of_matches] + [np.nan for i in range(7)])
            else:
                src_pts = np.float32([crop_kpts[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([painting_kpts[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                if M is None:
                    metrics.append([crop_path, painting_name, no_of_descriptors, no_of_matches, np.median(distances),
                                    np.mean(distances), np.std(distances)] + [np.nan for i in range(4)])
                else:
                    metrics.append([crop_path, painting_name, no_of_descriptors,
                                    no_of_matches,
                                    np.median(distances),
                                    np.mean(distances),
                                    np.std(distances),
                                    np.sum(mask == 0),
                                    np.sum(mask == 1),
                                    LA.norm(M, 'fro'),
                                    LA.det(M)])

metrics_df = pd.DataFrame(metrics, columns=['crop_path', 'painting_name', 'no_of_descriptors',
                                            'no_of_matches',
                                            'dist_median',
                                            'dist_mean',
                                            'dist_std',
                                            'no_of_outliers',
                                            'no_of_inliers',
                                            'homography_norm',
                                            'homography_det'])

metrics_df.to_csv("test.csv")
