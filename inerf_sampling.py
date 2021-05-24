import numpy as np
import torch
import cv2

def get_random_pixels(H, W, N_rand):
    '''
    randomly sample N_rand rays from all HxW possibilities
    '''
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)

    return select_coords

def get_interest_region_pixels(H, W, query, N_rand, save_path):
    '''
    use orb features to pick keypoints, create interest regions
    '''
    # max number of keypoints to detect
    orb_detector = cv2.ORB_create(nfeatures=(N_rand * 2))

    query_features = np.copy(query.cpu().numpy()) * 255
    query_features = cv2.cvtColor(query_features.astype(np.uint8), cv2.COLOR_RGB2BGR)
    keypoints = orb_detector.detect(query_features, None)

    key_regions = cv2.drawKeypoints(query_features, keypoints, None, color=(0, 0, 255))
    if save_path is not None:
        cv2.imwrite(save_path + '/key_regions.png', key_regions)

    mask = np.zeros((H, W)).astype("uint8")
    for kp in keypoints:
        mask[int(kp.pt[1]), int(kp.pt[0])] = 255
    kernel = np.ones((5,5))
    dil_iterations = 3
    for i in range(dil_iterations):
        mask = cv2.dilate(mask, kernel)

    if save_path is not None:
        cv2.imwrite(save_path + '/region_mask.png', mask)
    coords = np.argwhere(mask > 0)
    np.random.shuffle(coords)

    return coords
    