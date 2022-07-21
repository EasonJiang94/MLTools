import os
import numpy as np
from data import Data
import orjson
from PIL import Image
class Dataset(object):
    yolo_class_tabel = {
        0 : "human"
    }
    label_template = {
        "class" : "Undefined", 
        "xyxy" : [0,0,0,0],
        "xywh" : [0,0,0,0]
    }
    def __init__(self, name="Undefined"):
        self.name = name
        self.data_dict = {}
        self.bbox_cnt = 0
        self.class_list = []
        self.image_dir_list = []
        self.label_dir_list = []
    def _get_file_path_list(self, dir_path, extend=".jpg"):
        return [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith(extend)]
        
    def _path_to_path(self, path, dir_path, ext=".txt"):
        image_name = os.path.split(path)[1]
        label_path = os.path.join(dir_path, image_name)
        label_path = os.path.splitext(label_path)[0]
        label_path = label_path + ext
        return label_path
    
    def _parse_yolo_labels(self, label_path) -> list:
        label_list = []
        with open(label_path, "r") as yolo_file:
            for line in yolo_file:
                line = line.replace("\n","")
                c, x, y, w, h = line.split(' ')
                if c not in self._yolo_class_filter:
                    continue
                c = int(c)
                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)
                
                if (x + (w/2)) > 1:
                    w = (1 - x) * 2
                if (x - (w/2)) < 0:
                    w = (x - 0) * 2
                if (y + (h/2)) > 1:
                    h = (1 - y) * 2
                if (y - (h/2)) < 0:
                    h = (y - 0) * 2
                label = {}
                label["class"] = Dataset.yolo_class_tabel[c] if c in Dataset.yolo_class_tabel.keys() else str(c)
                label["xywh"] = (x, y, w, h)
                label_list.append(label)
        return label_list

    def _parse_labelme_labels(self, label_path):
        label_list = []
        with open(label_path, 'r') as f:
            labelme_data = orjson.loads(f.read())
            resolution = (labelme_data["imageWidth"], labelme_data["imageHeight"])
            for shape in labelme_data["shapes"]:
                label_class = shape["label"]
                if label_class in self._labelme_blacklist:
                    continue
                xyxy = self._get_xyxy_from_shape(shape['points'])
                label = {}
                label["class"] = label_class
                label["xyxy"] = xyxy
                label_list.append(label)
            image_path = labelme_data["imagePath"].replace("\\", "/")

        return image_path, label_list, resolution

    def _parse_v7_seg_labels(self, label_path):
        label_list = []
        with open(label_path, 'r') as f:
            seg_data = orjson.loads(f.read())
            images = seg_data.get("images")
            anns = seg_data.get("annotations")
            categories = seg_data.get("categories")
            return images, anns, categories

    def _get_xyxy_from_shape(self, points:list):
        xmin = 999999
        ymin = 999999
        xmax = -1
        ymax = -1
        for point in points:
            assert len(point) >= 2, 'Invalid point format'
            x, y = point
            if x < xmin:
                xmin = x
            if y < ymin:
                ymin = y
            if x > xmax:
                xmax = x
            if y > ymax:
                ymax = y
        return (xmin, ymin, xmax, ymax)
            
    def import_from_yolo(self, image_dir, label_dir, attribute=None, class_filter = [], set_ir_flag=False):
        ### need to update parameters: self.bbox_cnt, self.class_list, self.data_dict ###
        self.image_dir_list.append(image_dir)
        self.label_dir_list.append(label_dir)
        self._yolo_class_filter = class_filter
        image_path_list = self._get_file_path_list(image_dir, extend=".jpg")
        local_img_cnt = 0
        local_bbox_cnt = 0
        for image_path_idx, image_path in enumerate(image_path_list):
            print(f"import_from_yolo : {image_path_idx / len(image_path_list)*100:2.2f}%", end='\r')
            image_name = os.path.split(image_path)[1]
            label_path = self._path_to_path(image_path, label_dir, ext=".txt")
            if image_name in self.data_dict:
                print(f"find duplicated data : {image_name}")
                continue
            data = Data(image_name)
            data.set_image_dir(image_dir)
            local_img_cnt  += 1
            if os.path.exists(label_path):
                label_list = self._parse_yolo_labels(label_path)
                resolution = Image.open(image_path).size # w, h
                data.set_resolution(resolution)
                self.bbox_cnt += len(label_list)
                local_bbox_cnt += len(label_list)
                for label in label_list:
                    assert "xywh" in label, "no xywh in label"
                    data.add_by_xywh(label["xywh"], class_name=label["class"], attribute=None)
                    if label["class"] in self.class_list:
                        pass
                    else : 
                        self.class_list.append(label["class"])
                pass
            else:
                pass
            if set_ir_flag:
                data.set_ir_flag()
            self.data_dict[image_name] = data
            # print(data)
        print(f"import_from_yolo : 100%, has {local_img_cnt} images and {local_bbox_cnt} bboxes")

    def import_from_labelme(self, image_dir, label_dir, attribute=None, set_ir_flag=False, blacklist = []):
        ### need to update parameters: self.bbox_cnt, self.class_list, self.data_dict ###
        self.image_dir_list.append(image_dir)
        self.label_dir_list.append(label_dir)
        self._labelme_blacklist = blacklist
        label_path_list = self._get_file_path_list(label_dir, extend=".json")
        local_bbox_cnt = 0
        local_img_cnt = 0
        for label_path_idx, label_path in enumerate(label_path_list):
            print(f"import_from_labelme : {label_path_idx / len(label_path_list)*100:2.2f}%", end='\r')
            # image_path = self._path_to_path(label_path, image_dir, ext=".jpg")
            if os.path.exists(label_path):
                image_path, label_list, resolution = self._parse_labelme_labels(label_path)
                image_name = os.path.split(image_path)[1]
                # print(f"{image_path = }\n{image_name = }")
                if image_name in self.data_dict:
                    print(f"find duplicated data : {image_name}")
                    continue
                data = Data(image_name)
                data.set_image_dir(image_dir)
                local_img_cnt += 1
                data.set_resolution(resolution)
                self.bbox_cnt += len(label_list)
                local_bbox_cnt += len(label_list)
                for label in label_list:
                    assert "xyxy" in label, "no xyxy in label"
                    data.add_by_xyxy(label["xyxy"], class_name=label["class"], attribute=None, ratio=False)
                    if label["class"] in self.class_list:
                        pass
                    else : 
                        self.class_list.append(label["class"])
                    # print(data)

            else:
                # print(f"{image_path = }, \n{label_path = }")
                pass
            if set_ir_flag:
                data.set_ir_flag()
            self.data_dict[image_name] = data
        print(f"import_from_labelme : 100.00%, has {local_img_cnt} images and {local_bbox_cnt} bboxes")

    def import_from_v7_segmentation(self, image_dir, label_dir, attribute=None, set_ir_flag=True, blacklist = []):
        ### Due to the format of v7 segmentation is very strange ###
        ### The parsing flow is totally different with others    ###
        ### need to update parameters: self.bbox_cnt, self.class_list, self.data_dict ###
        self.image_dir_list.append(image_dir)
        self.label_dir_list.append(label_dir)
        self._labelme_blacklist = blacklist
        label_path_list = self._get_file_path_list(label_dir, extend=".json")
        local_segmentation_cnt = 0
        local_img_cnt = 0
        for label_path_idx, label_path in enumerate(label_path_list):
            images, anns, categories = self._parse_v7_seg_labels(label_path)
            for image_idx, image_info in enumerate(images):
                print(f"import_from_v7 : {image_idx / len(images)*100:2.2f}%", end='\r')
                image_name = image_info.get("file_name")
                image_id = image_info.get("id")
                image_width = image_info.get("width", 0)
                image_height = image_info.get("height", 0)
                resolution = (image_width, image_height)
                if image_name is None:
                    print(f"got file_name FAIL : {image_idx = }")
                    continue
                image_path = os.path.join(image_dir, image_name)
                data = Data(image_name)
                data.set_image_dir(image_dir)
                data.set_resolution(resolution)
                if set_ir_flag:
                    data.set_ir_flag()
                local_img_cnt += 1
                ann_for_this_img = (ann for ann in anns if ann.get("image_id") == image_id)

                for ann in ann_for_this_img:
                    seg = ann.get("segmentation", [])
                    bbox = ann.get("bbox", [0,0,0,0])
                    attributes = ann.get("extra", {}).get("attributes", [])
                    category_id = ann.get("ann_for_this_img", 1)
                    class_name = categories[category_id -1].get("name")
                    # print(f"{class_name = }, {attribute = }, {bbox = }")
                    data.add_by_seg(seg, bbox=bbox, attributes=attributes, class_name=class_name)
                    local_segmentation_cnt += 1
                    self.data_dict[image_name] = data
                    if class_name in self.class_list:
                        pass
                    else : 
                        self.class_list.append(class_name)

                # print(f"{image_idx = }, {len(list(ann_for_this_img))}")
        self.bbox_cnt += local_segmentation_cnt
        print(f"import_from_v7_segmentation : 100.00%, has {local_img_cnt} images and {local_segmentation_cnt} segmentations")

    def __repr__(self):
        return (f"Dataset name : {self.name}" \
        f"\nClasses list : {self.class_list}" \
        f"\nAmount of images : {len(self.data_dict)}" \
        f"\nAmount of bounding boxes : {self.bbox_cnt}" \
        f"\nSource image directories : \n{self.image_dir_list}" \
        f"\nSource label directories : \n{self.label_dir_list}")
