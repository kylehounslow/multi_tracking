"""
Some computer vision utility functions
"""
import base64, cv2, os, glob
import numpy as np
import math


def resize_pad_image(img, new_dims, pad_output=True):
    old_height, old_width, ch = img.shape
    old_ar = float(old_width) / float(old_height)
    new_ar = float(new_dims[0]) / float(new_dims[1])
    undistorted_scale_factor = [1.0, 1.0]  # if you want to resize bounding boxes on a padded img you'll need this
    if pad_output is True:
        if new_ar > old_ar:
            new_width = old_height * new_ar
            padding = abs(new_width - old_width)
            img = cv2.copyMakeBorder(img, 0, 0, 0, int(padding), cv2.BORDER_CONSTANT, None, [0, 0, 0])
            undistorted_scale_factor = [float(old_width) / (float(new_dims[1]) * old_ar),
                                        float(old_height) / float(new_dims[1])]
        elif new_ar < old_ar:
            new_height = old_width / new_ar
            padding = abs(new_height - old_height)
            img = cv2.copyMakeBorder(img, 0, int(padding), 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])
            undistorted_scale_factor = [float(old_width) / float(new_dims[0]),
                                        float(old_height) / (float(new_dims[0]) / old_ar)]
        elif new_ar == old_ar:
            scale_factor = float(old_width) / new_dims[0]
            undistorted_scale_factor = [scale_factor, scale_factor]
    outimg = cv2.resize(img, (new_dims[0], new_dims[1]))
    return outimg, undistorted_scale_factor


def crop_img(bbox, im):
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    cropped_img = im[y1:y2, x1:x2]
    return cropped_img


def compute_dist(vec1, vec2, mode='cosine'):
    """
    compute the distance between two given vectors.
    :param vec1: np.array vector
    :param vec2: np.array vector
    :param mode: cosine for cosine distance; l2 for l2 norm distance;
    :return: distance of the input mode
    """
    if mode == 'cosine':
        dist = 1 - np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
    elif mode == 'l2':
        dist = np.linalg.norm(vec1 - vec2)
    else:
        dist = None
    return dist


def make_grids_of_images_from_folder(images_path, image_shape, grid_shape):
    """
    makes grids of images in numpy array format from an image folder.

    :param images_path: string, path to images folder
    :param image_shape: tuple, size each image will be resized to for display
    :param grid_shape: tuple, shape of image grid (rows,cols)
    :return: list of grid images in numpy array format

    example usage: grids = make_grids_of_images('/Pictures', (64,64),(5,5))

    """
    # get all images from folder
    img_path_glob = glob.iglob(os.path.join(images_path, '*'))
    img_path_list = []
    for ip in img_path_glob:
        if ip.endswith('.jpg') or ip.endswith('.jpeg') or ip.endswith('.png'):
            img_path_list.append(ip)
    if len(img_path_list) < 1:
        print 'No images found at {}'.format(images_path)
        return None
    image_grids = []
    # start with black canvas to draw images to
    grid_image = np.zeros(shape=(image_shape[1] * (grid_shape[1]), image_shape[0] * grid_shape[0], 3),
                          dtype=np.uint8)
    cursor_pos = [0, 0]
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        if img is None:
            print 'ERROR: reading {}. skipping.'.format(img_path)
            continue
        img = cv2.resize(img, image_shape)
        # draw image to black canvas
        grid_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= grid_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= grid_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                # reset black canvas
                grid_image = np.zeros(shape=(image_shape[1] * (grid_shape[1]), image_shape[0] * grid_shape[0], 3),
                                      dtype=np.uint8)
        image_grids.append(grid_image)

    return image_grids


def make_grids_of_images_from_list(image_list, image_shape, grid_shape):
    """
    makes grids of images in numpy array format from an image folder.

    :param images_path: list, input images
    :param image_shape: tuple, size each image will be resized to for display
    :param grid_shape: tuple, shape of image grid (rows,cols)
    :return: list of grid images in numpy array format

    example usage: grids = make_grids_of_images('/Pictures', (64,64),(5,5))

    """
    image_grids = []
    # start with black canvas to draw images to
    grid_image = np.zeros(shape=(image_shape[1] * (grid_shape[1]), image_shape[0] * grid_shape[0], 3),
                          dtype=np.uint8)
    cursor_pos = [0, 0]
    for img in image_list:
        img = cv2.resize(img, image_shape)
        # draw image to black canvas
        grid_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= grid_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= grid_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                # reset black canvas
                grid_image = np.zeros(shape=(image_shape[1] * (grid_shape[1]), image_shape[0] * grid_shape[0], 3),
                                      dtype=np.uint8)
        image_grids.append(grid_image)

    return image_grids


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def centroid_from_bb(bb):
    x1, y1, x2, y2 = bb
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    c_x = x1 + w / 2
    c_y = y1 + h / 2

    return np.array([c_x, c_y])


def dist_btwn_bb_centroids(bb1, bb2):
    dx, dy = centroid_from_bb(bb1) - centroid_from_bb(bb2)
    dist = math.sqrt(dx * dx + dy * dy)
    return dist


def wid_ht_from_bb(bb):
    wid = int(abs(bb[2] - bb[0]))
    ht = int(abs(bb[3] - bb[1]))
    return wid, ht


def check_tracks_equal(track1, track2):
    t1_bb = track1.get_latest_bb()
    t2_bb = track2.get_latest_bb()
    dist = np.linalg.norm(t2_bb - t1_bb)
    return dist < 50


def clamp_negative_nums(bb):
    temp = []
    for pnt in bb:
        tmp = pnt
        if tmp < 0:
            tmp = 0
        temp.append(tmp)
    return temp


def bb_has_width_height(bb):
    w = int(bb[2] - bb[0])
    h = int(bb[3] - bb[1])
    return True if (w > 1 and h > 1) else False


def bb_as_ints(bb):
    return [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
