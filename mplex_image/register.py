import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import transform, util
from skimage import data, img_as_float
from skimage.util import img_as_ubyte
import cv2
import sys

# code from adapted chandler gatenbee and brian white
# https://github.com/IAWG-CSBC-PSON/registration-challenge

def match_keypoints(moving, target, feature_detector):
    '''
    :param moving: image that is to be warped to align with target image
    :param target: image to which the moving image will be aligned
    :param feature_detector: a feature detector from opencv
    :return:
    '''

    kp1, desc1 = feature_detector.detectAndCompute(moving, None)
    kp2, desc2 = feature_detector.detectAndCompute(target, None)

    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    src_match_idx = [m.queryIdx for m in matches]
    dst_match_idx = [m.trainIdx for m in matches]

    src_points = np.float32([kp1[i].pt for i in src_match_idx])
    dst_points = np.float32([kp2[i].pt for i in dst_match_idx])

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransacReprojThreshold=10)

    good = [matches[i] for i in np.arange(0, len(mask)) if mask[i] == [1]]

    filtered_src_match_idx = [m.queryIdx for m in good]
    filtered_dst_match_idx = [m.trainIdx for m in good]

    filtered_src_points = np.float32([kp1[i].pt for i in filtered_src_match_idx])
    filtered_dst_points = np.float32([kp2[i].pt for i in filtered_dst_match_idx])

    return filtered_src_points, filtered_dst_points

def apply_transform(moving, target, moving_pts, target_pts, transformer, output_shape_rc=None):
    '''
    :param transformer: transformer object from skimage. See https://scikit-image.org/docs/dev/api/skimage.transform.html for different transformations
    :param output_shape_rc: shape of warped image (row, col). If None, uses shape of traget image
    return
    '''
    if output_shape_rc is None:
        output_shape_rc = target.shape[:2]

    if str(transformer.__class__) == "<class 'skimage.transform._geometric.PolynomialTransform'>":
        transformer.estimate(target_pts, moving_pts)
        warped_img = transform.warp(moving, transformer, output_shape=output_shape_rc)

        ### Restimate to warp points
        transformer.estimate(moving_pts, target_pts)
        warped_pts = transformer(moving_pts)
    else:
        transformer.estimate(moving_pts, target_pts)
        warped_img = transform.warp(moving, transformer.inverse, output_shape=output_shape_rc)
        warped_pts = transformer(moving_pts)

    return warped_img, warped_pts

def keypoint_distance(moving_pts, target_pts, img_h, img_w):
    dst = np.sqrt(np.sum((moving_pts - target_pts)**2, axis=1)) / np.sqrt(img_h**2 + img_w**2)
    return np.mean(dst)




def register(target_file,moving_file, b_plot=False):
    s_round = moving_file.split('_')[0]
    s_sample = moving_file.split('_')[2]
    print(s_round)
    target = img_as_ubyte(img_as_float(Image.open(target_file)))
    moving = img_as_ubyte(img_as_float(Image.open(moving_file)))
    
    fd = cv2.AKAZE_create()
    #fd = cv2.KAZE_create(extended=True)
    moving_pts, target_pts = match_keypoints(moving, target, feature_detector=fd)

    transformer = transform.SimilarityTransform()
    warped_img, warped_pts = apply_transform(moving, target, moving_pts, target_pts, transformer=transformer)

    warped_img = img_as_ubyte(warped_img)
    
    print("Unaligned offset:", keypoint_distance(moving_pts, target_pts, moving.shape[0], moving.shape[1]))
    print("Aligned offset:", keypoint_distance(warped_pts, target_pts, moving.shape[0], moving.shape[1]))
    if b_plot:
        fig, ax = plt.subplots(2,2, figsize=(10,10))
        ax[0][0].imshow(target)
        ax[0][0].imshow(moving, alpha=0.5)
        ax[1][0].scatter(target_pts[:,0], -target_pts[:,1])
        ax[1][0].scatter(moving_pts[:,0], -moving_pts[:,1])

        ax[0][1].imshow(target)
        ax[0][1].imshow(warped_img, alpha=0.5)
        ax[1][1].scatter(target_pts[:,0], -target_pts[:,1])
        ax[1][1].scatter(warped_pts[:,0], -warped_pts[:,1])
        plt.savefig(f"../../QC/RegistrationPlots/{s_sample}_{s_round}_rigid_align.png", format="PNG")
    return(moving_pts, target_pts, transformer) 
