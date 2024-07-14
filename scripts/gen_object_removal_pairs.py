import cv2
import numpy as np


def get_central_coord(mask):
    non_zero_coords = cv2.findNonZero(mask)
    coords = non_zero_coords.reshape(-1, 2)
    if len(coords) > 0:
        center_x = int(np.mean(coords[:, 0]))
        center_y = int(np.mean(coords[:, 1]))
    else:
        center_x, center_y = 0, 0
    return center_x, center_y


def gen_object_removal_pairs(src, mask, inpainted_bgr, bgr_mask):
    assert src.shape == mask.shape

    _, mask = cv2.threshold(mask , 127, 255, cv2.THRESH_BINARY)
    mask = mask / 255
    object = src * mask
    o_x, o_y = get_central_coord(object)
    h, w = object.shape[:2]

    b_x, b_y = get_central_coord(bgr_mask)

    shift_mat = np.float32([1, 0, b_x - o_x], [0, 1, b_y - o_y])
    tilde_o = cv2.warpAffine(object, shift_mat, (w, h))
    tilde_m = cv2.warpAffine(mask, shift_mat, (w, h))

    tiled_I = inpainted_bgr * (np.ones(tilde_m.shape) - tilde_m) + tilde_o
    return tilde_o, tilde_m, tiled_I


if __name__ == '__main__':
    src = cv2.imread("./imgs/sheep/input.png")
    mask = cv2.imread("./imgs/sheep/mask.png")
    background_mask = cv2.imread("./imgs/sheep/backgounrd.png")
    tilde_o, tilde_m, tiled_I = gen_object_removal_pairs(src, mask, src, background_mask)


