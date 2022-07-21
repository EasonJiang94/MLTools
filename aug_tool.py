import os
from dataset import Dataset
from data import Data
import cv2

class AugmentationTool(object):
    def __init__(self):
        self.segmentation = Dataset("Segmentation Dataset")
        self.background = Dataset("Background Dataset")

    def add_yolo_dataset(self, img_dir, label_dir, background=False, segmentation=False, attribute=None):
        if background:
            self.background.import_from_yolo(img_dir, label_dir, attribute=attribute, set_ir_flag=True, class_filter=["0"])
        if segmentation:
            self.segmentation.import_from_yolo(img_dir, label_dir, attribute=attribute, class_filter=["0"])
        
    def add_labelme_dataset(self, img_dir, label_dir, background=False, segmentation=False, attribute=None, blacklist=[]):
        if background:
            self.background.import_from_labelme(img_dir, label_dir, attribute=attribute, set_ir_flag=True, blacklist=blacklist)
        if segmentation:
            self.segmentation.import_from_labelme(img_dir, label_dir, attribute=attribute, set_ir_flag=False)

    def add_v7_segmentation_dataset(self, img_dir, label_dir, background=False, segmentation=False, attribute=None, blacklist=[]):
        if background:
            self.background.import_from_v7_segmentation(img_dir, label_dir, attribute=attribute, set_ir_flag=True, blacklist=blacklist)
        if segmentation:
            self.segmentation.import_from_v7_segmentation(img_dir, label_dir, attribute=attribute, set_ir_flag=False)
    def _pick_one_segmentation_randomly(self, class_name=None, attributes=None):
        pass
    def __repr__(self):
        return f"Segmentation Dataset : \n{self.segmentation}\nBackground Dataset : \n{self.background}\n"

if __name__=="__main__":
    aug_tool = AugmentationTool()
    auo_seg_img_dir = "/mnt/220/labelled_data/copy-paste-aug/segmentation/images/train_0115"
    auo_seg_label_dir = "/mnt/220/labelled_data/copy-paste-aug/segmentation/annotaions_20210115"
    aug_tool.add_v7_segmentation_dataset(auo_seg_img_dir, auo_seg_label_dir, segmentation=True)
    print(aug_tool)