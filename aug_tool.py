import os
from dataset import Dataset
from data import Data
import cv2
from utils import make_if_inexist_recursive, make_if_inexist, colorjitter, image_rotate
import numpy as np
import random
import time
from torch import unique
from torchvision.ops import masks_to_boxes
from torch import tensor
class AugmentationTool(object):
    def __init__(self, output_dir=None):
        self.segmentation = Dataset("Segmentation Dataset")
        self.background = Dataset("Background Dataset")
        self.output_dir = output_dir
        self.cnt = 0
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

    def _get_mask_by_labels(self, img, labels):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for label in labels:
            seg = label.get("seg")
            # print(f"{seg = }")
            if not seg is None:
                seg = np.array(seg, dtype=np.int32)
                cv2.fillPoly(mask,[seg],1)
            else:
                xmin = label.get("xmin", 0)
                ymin = label.get("ymin", 0)
                xmax = label.get("xmax", 0)
                ymax = label.get("ymax", 0)
                mask[ymin:ymax, xmin:xmax] = 1
        return mask

    def paste_segmentation_to_background(self, seg_per_image_limit=15, background_limit=1E8):
        make_if_inexist(self.output_dir)
        for data_idx, data in enumerate(self.background): ## get a Data object from Dataset
            background_image = data.get_image()
            # print(data.labels)
            background_mask = self._get_mask_by_labels(background_image, data.labels)
            output_path = os.path.join(self.output_dir, data.image_name)
            cv2.imwrite(output_path, background_mask)
            print(f"{background_mask.shape = }, {output_path = }")
            random_segs = self._pick_segmentation_randomly(attributes=["over50", "with_head"], n=25)
            # print(len(random_segs))
            output_path = os.path.join(self.output_dir, f"before_{data.image_name}")
            cv2.imwrite(output_path, background_image)
            for seg_idx, seg in enumerate(random_segs):
                
                seg_mask = seg["mask"]
                seg_image = seg["image"]
                len_mask = seg_mask.shape[0] # shape of mask must be square
                # print(f"{data.resolution = }")
                max_paste_x = data.resolution[0] - len_mask -1
                max_paste_y = data.resolution[1] - len_mask -1
                try_time = 0
                while(True):
                    if try_time > 20:
                        print("[paste_segmentation_to_background] I Can't paste segmentation into image")
                        break
                    random.seed(int(time.time()*1000000), version=2)
                    rand_xmin = random.randint(0, max_paste_x)
                    rand_ymin = random.randint(0, max_paste_y)
                    rand_xmax = rand_xmin + len_mask
                    rand_ymax = rand_ymin + len_mask
                    # print(f"{rand_xmin = }")
                    # print(f"{rand_xmax = }")
                    # print(f"{rand_ymin = }")
                    # print(f"{rand_ymax = }")
                    # print(f"{len_mask = }")
                    # print(f"{background_mask.shape = }")
                    
                    and_mask = np.logical_and(background_mask[rand_ymin:rand_ymax, rand_xmin:rand_xmax], seg_mask)
                    if np.sum(and_mask, dtype=np.int32) == 0:
                        background_mask[rand_ymin:rand_ymax, rand_xmin:rand_xmax] = np.logical_or(background_mask[rand_ymin:rand_ymax, rand_xmin:rand_xmax], seg_mask)
                        crop_region = background_image[rand_ymin:rand_ymax, rand_xmin:rand_xmax]
                        crop_region[seg_mask==1] = 0
                        crop_region = crop_region + seg_image
                        background_image[rand_ymin:rand_ymax, rand_xmin:rand_xmax] = crop_region
                        ### starting hard code here eason.jiang@beseye.com ###
                        mask_tensor = tensor(seg_mask)
                        obj_ids = unique(mask_tensor)
                        # first id is the background, so remove it.
                        obj_ids = obj_ids[1:]
                        # split the color-encoded mask into a set of boolean masks.
                        # Note that this snippet would work as well if the masks were float values instead of ints.
                        masks = mask_tensor == obj_ids[:, None, None]
                        boxes = masks_to_boxes(masks).numpy()
                        xmin = 99999
                        ymin = 99999
                        xmax = -1
                        ymax = -1
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            x1 = int(x1)
                            y1 = int(y1)
                            x2 = int(x2)
                            y2 = int(y2)
                            if xmin > x1:
                                xmin = x1
                            if xmax < x2:
                                xmax = x2
                            if ymin > y1:
                                ymin = y1
                            if ymax < y2:
                                ymax = y2
                        xmin = xmin + rand_xmin
                        ymin = ymin + rand_ymin
                        xmax = xmax + rand_xmin
                        ymax = ymax + rand_ymin
                        xyxy = (xmin, ymin, xmax, ymax)
                        # cv2.rectangle(background_image, (xmin, ymin), (xmax, ymax), (30, 30, 220), 2, cv2.LINE_AA)
                        self.background.data_dict[data.image_name].add_by_xyxy(xyxy, ratio=False, class_name="Human", attributes="from_seg")
                        self.background.data_dict[data.image_name].set_image(background_image)
                        break                    
                    # print(f"[paste_segmentation_to_background] {output_path} fail to paste: {try_time}")
                    # print(f"{np.sum(and_mask, dtype=np.int32) = }")
                    try_time += 1
                
            output_path = os.path.join(self.output_dir, data.image_name)
            cv2.imwrite(output_path, background_mask*255)
            print(f"{np.max(background_mask) = }")
                

            output_path = os.path.join(self.output_dir, f"after_{data.image_name}")
            cv2.imwrite(output_path, background_image)
                
                
                


    def _pick_segmentation_randomly(self, n=1, \
                                    class_name=None, \
                                    attributes=None, \
                                    w_limit=(0.028, 1.0), \
                                    h_limit=(0.05, 1.0), \
                                    jitter=True, \
                                    rotate=True):
        seg_list = [] #[{"mask" = np.array, "image" = np.array}]
        random.seed(int(time.time()*1000000), version=2)
        random_data = self.segmentation.get_random_data(n=n)
        
        for data_idx, data in enumerate(random_data):
            # print(f"{data.image_name = }")
            labels = [i for i in data.labels if 
                    i["w"] > w_limit[0] and 
                    i["w"] < w_limit[1] and 
                    i["h"] > h_limit[0] and 
                    i["h"] < h_limit[1] and 
                    set(attributes).issubset(set(i["attributes"]))] 
            pick_times = 0
            while(len(labels) <= 0):
                if pick_times > 1000:
                    print("[_pick_segmentation_randomly] ERROR: it's too hard to find data, remove the wh limit")
                    w_limit = (0.0, 1.0)
                    h_limit = (0.0, 1.0)
                random.seed(int(time.time()*1000000), version=2)
                data = self.segmentation.get_random_data(n=1)[0]
                labels = [i for i in data.labels if 
                    i["w"] > w_limit[0] and 
                    i["w"] < w_limit[1] and 
                    i["h"] > h_limit[0] and 
                    i["h"] < h_limit[1] and
                    set(attributes).issubset(set(i["attributes"]))] 
                pick_times += 1
            random.seed(int(time.time()*1000000), version=2)
            random_label = random.choice(labels)
            # print(f"{random_label['w']*100 = }")
            image = data.get_image()
            label_w = random_label["xmax"] - random_label["xmin"]
            label_h = random_label["ymax"] - random_label["ymin"]
            square_len = int(max(label_w, label_h)*1.41421356) # 1.41421356 means sqrt(2)
            square_mask = np.zeros((square_len, square_len), dtype=np.uint8)
            square_image = np.zeros((square_len, square_len, 3), dtype=np.uint8)
            mask = self._get_mask_by_labels(image, [random_label])
            
            start_x = int((square_len - (label_w)) / 2)
            start_y = int((square_len - (label_h)) / 2)
            try:
                square_mask[start_y:start_y+label_h, start_x:start_x+label_w] = mask[random_label["ymin"]:random_label["ymax"], random_label["xmin"]:random_label["xmax"]]
                square_image[start_y:start_y+label_h, start_x:start_x+label_w, :] = image[random_label["ymin"]:random_label["ymax"], random_label["xmin"]:random_label["xmax"], :]
            except:
                continue
            if jitter:
                random_value = random.randint(1, 10)
                if random_value <= 2:
                    square_image = colorjitter(square_image, cj_type="b")
                random_value = random.randint(1, 10)
                if random_value <= 2:
                    square_image = colorjitter(square_image, cj_type="s")
                random_value = random.randint(1, 10)
                if random_value <= 2:
                    square_image = colorjitter(square_image, cj_type="c")
                
            if rotate:
                random_value = random.randint(1, 2)
                if random_value <=1:
                    random_value = random.randint(-180, 180)
                    square_image = image_rotate(square_image, angle=random_value)
                    square_mask = image_rotate(square_mask, angle=random_value)
            
            square_image =  np.multiply(square_image, cv2.cvtColor(square_mask,cv2.COLOR_GRAY2BGR))
            self.cnt+=1
            seg = {}
            seg["mask"] =  square_mask
            seg["image"] = square_image
            seg_list.append(seg)
            # output_path = os.path.join(self.output_dir, f"{self.cnt:03d}.jpg")
            # cv2.imwrite(output_path, mask*255)
            # output_path = os.path.join(self.output_dir, f"square_mask_{self.cnt:03d}.jpg")
            # cv2.imwrite(output_path, square_mask*255)
            # output_path = os.path.join(self.output_dir, f"square_image_{self.cnt:03d}.jpg")
            # cv2.imwrite(output_path, square_image)
        return seg_list


    def __repr__(self):
        return f"Segmentation Dataset : \n{self.segmentation}\nBackground Dataset : \n{self.background}\n"

if __name__=="__main__":
    aug_tool = AugmentationTool(output_dir="test")
    auo_seg_img_dir = "/mnt/220/labelled_data/copy-paste-aug/segmentation/images/train_0115"
    auo_seg_label_dir = "/mnt/220/labelled_data/copy-paste-aug/segmentation/annotaions_20210115"
    aug_tool.add_v7_segmentation_dataset(auo_seg_img_dir, auo_seg_label_dir, segmentation=True)
    yolov5_image_dir = "/mnt/TrainingMachine/HD_training_data/train_KintetsuPOC/images/val"
    yolov5_label_dir = "/mnt/TrainingMachine/HD_training_data/train_KintetsuPOC/labels/val"
    aug_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, background=True)

    print(aug_tool)
    aug_tool.paste_segmentation_to_background()
    output_dir = "output_test"
    make_if_inexist(output_dir)
    aug_tool.background.export_yolo_format(output_dir)