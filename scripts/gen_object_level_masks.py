import os
import sys
import cv2
import numpy as np

sys.path.append("/your/mask2former/path")
from algorithms.mask2former.predictor import setup_cfg, VisualizationDemo
from algorithms.mask2former.maskformer_model import Maskformer

model_path = "/your/mask2former/ckpt/path"
# yaml config is from Mask2fomer
cfg = setup_cfg(config_file="configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml")
panoptic_demo = VisualizationDemo(cfg)

with open('coco_labels.txt', 'r') as f:
    names = f.readlines()
    names = [it.strip() for it in names]

with open('coco_object_labels.txt', 'r') as f:
    object_names = f.readlines()
    object_names = [it.strip() for it in object_names]



def is_object_cls(cls_name):
    status = False
    for object_name in object_names:
        if cls_name in object_name:
            status = True
            break
    return status


def gen_object_level_masks(img, mask_save_path, save_visual_demo_path=None):
    '''
    img:np.array, RGB
    '''
    predictions, visualizaed_output = panoptic_demo.run_on_image(img[:,:, ::-1])
    if save_visual_demo_path is not None:
        visualizaed_output.save(save_visual_demo_path)

    panoptic_seg, segments_info = predictions["panoptic_seg"]
    panoptic_seg = panoptic_seg.cpu().numpy()
    all_objects_mask = np.zeros_like(panoptic_seg)
    for msk_info in segments_info:
        cls_name = names[msk_info['category_id']]
        cls_name = cls_name.split('-')[0]
        if not is_object_cls(cls_name):
            all_objects_mask = all_objects_mask + msk_arr
        msk = panoptic_seg == msk_info['id']
        msk_arr = msk * 255
        cv2.imwrite(os.path.join(mask_save_path, f"mask_{cls_name}.png"), msk_arr)

    all_objects_mask[all_objects_mask > 0] = 255
    cv2.imwrite(os.path.join(mask_save_path, f".png"), all_objects_mask)



if __name__ == '__main__':
    img = cv2.imread('./imgs/sheep/input.png')
    gen_object_level_masks(img, "./imgs/sheep/")

