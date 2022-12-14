from copy import deepcopy as dcp
import cv2
from numpy import array_equal
import numpy as np
import os
class Data(object):
    idx = 0
    label_template = {
        "class_name" : "human",
        "attributes" : [],
        "x" : 0.0,
        "y" : 0.0,
        "w" : 0.0,
        "h" : 0.0,
        "xmin" : 0.0,        
        "ymin" : 0.0,
        "xmax" : 0.0,
        "ymax" : 0.0,        
        "seg" : None,
        "c" : 1.0 ## means confidence score, conf of gt always be 1.0
    }
    def __init__(self, image_name="Unknown"):
        self.image_name = image_name
        self.image_dir = "Unknown"
        self.labels = []
        self.ir = None
        self.cam_name = "Unknown"
        self.resolution = None # w, h
        self._image = None
        
    @property
    def label_num(self) ->int:
        return len(self.labels)

    @property
    def image_path(self):
        assert os.path.exists(self.image_dir), f"image dir is not inexist : {self.image_dir}"
        image_path = os.path.join(self.image_dir, self.image_name)
        if not os.path.exists(image_path):
            print(f"Warning! image path is not exist : {image_path}")
        return image_path

    def set_image_dir(self, image_dir:str):
        assert os.path.exists(image_dir), f"image dir is not inexist : {image_dir}"
        self.image_dir = image_dir

    def set_resolution(self, w:int, h:int):
        self.resolution = (w,h)

    def set_resolution(self, resolution:tuple):
        assert len(resolution)==2, 'size of resolution must be 2'
        assert isinstance(resolution[0], int), 'element of resolution must be int'
        assert isinstance(resolution[1], int), 'element of resolution must be int'
        self.resolution = resolution

    def _read_image(self):
        if not self._image is None:
            return True
        assert os.path.exists(self.image_dir), f"image dir is not inexist : {self.image_dir}"
        image_path = os.path.join(self.image_dir, self.image_name)
        if not os.path.exists(image_path):
            print(f"Warning! image path is not exist : {image_path}")
            return False
        self._image = cv2.imread(self.image_path)
        if self._image is None:
            print(f"Warning! image is none : {self.image_path}")
            return False
        return True
            
        
    def set_ir_flag(self):
        if not self._read_image():
            return False
        if array_equal(self._image[:,:,0], self._image[:, :, 1]):
            self.ir = True
        else:
            self.ir = False
        return True

    def set_image(self, image):
        self.image = image

    def get_image(self):
        if self._read_image():
            return self._image
        else:
            return None

    def free_image(self):
        del(self._image)
        self._image = None

    def add_by_xyxy(self, xyxy:list, \
                    ratio=True, \
                    conf=1.0, \
                    class_name=None, \
                    attributes=None):
        xmin = xyxy[0]
        ymin = xyxy[1]
        xmax = xyxy[2]
        ymax = xyxy[3]
        
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        w = (xmax - xmin)
        h = (ymax - ymin)

        label = dcp(Data.label_template)
        if class_name:
            label["class_name"] = class_name
        if attributes:
            label["attributes"] = attributes

        if ratio:
            label["x"] = x
            label["y"] = y
            label["w"] = w
            label["h"] = h
            label["xmin"] = int(xmin * self.resolution[0])
            label["ymin"] = int(ymin * self.resolution[1])
            label["xmax"] = int(xmax * self.resolution[0])
            label["ymax"] = int(ymax * self.resolution[1])
        else:
            if self.resolution:
                label["x"] = x / self.resolution[0]
                label["y"] = y / self.resolution[1]
                label["w"] = w / self.resolution[0]
                label["h"] = h / self.resolution[1]
                label["xmin"] = int(xmin)
                label["ymin"] = int(ymin)
                label["xmax"] = int(xmax)
                label["ymax"] = int(ymax)
            else:
                print("[Data] ERROR, no self.resolution info")

        label["c"] = conf
        self.labels.append(label)
                  
    def add_by_xywh(self, xywh:list, \
                    ratio=True, \
                    conf=1.0, \
                    class_name=None, \
                    attributes=None):
        
        

        label = dcp(Data.label_template)
        if class_name:
            label["class_name"] = class_name
        if attributes:
            label["attributes"] = attributes
        x, y, w, h = xywh
        xmin = x - (w / 2)
        ymin = y - (h / 2)
        xmax = x + (w / 2)
        ymax = y + (h / 2)
        if ratio:
            label["x"] = x
            label["y"] = y
            label["w"] = w
            label["h"] = h
            label["c"] = conf
            label["xmin"] = int(xmin * self.resolution[0])
            label["ymin"] = int(ymin * self.resolution[1])
            label["xmax"] = int(xmax * self.resolution[0])
            label["ymax"] = int(ymax * self.resolution[1])
        else:
            if self.resolution:
                label["x"] = x / self.resolution[0]
                label["y"] = y / self.resolution[1]
                label["w"] = w / self.resolution[0]
                label["h"] = h / self.resolution[1]
                label["xmin"] = int(xmin)
                label["ymin"] = int(ymin)
                label["xmax"] = int(xmax)
                label["ymax"] = int(ymax)
            else:
                print("[Data] ERROR, no resolution info")
                return
        self.labels.append(label)

    def add_by_seg(self, seg:list, \
                    bbox=None, \
                    ratio=False, \
                    conf=1.0, \
                    class_name=None, \
                    attributes=None):
        label = dcp(Data.label_template)
        if attributes:
            label["attributes"] = attributes
        if bbox:
            x, y, w, h = bbox
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            xmin = x
            ymin = y
            xmax = x + w 
            ymax = y + h 
        else:
            
            xmin = min(seg[:,0])
            xmax = max(seg[:,0])
            ymin = min(seg[:,1])
            ymax = max(seg[:,1])

        if ratio:
            label["x"] = x
            label["y"] = y
            label["w"] = w
            label["h"] = h
            label["c"] = conf
            label["xmin"] = int(xmin * self.resolution[0])
            label["ymin"] = int(ymin * self.resolution[1])
            label["xmax"] = int(xmax * self.resolution[0])
            label["ymax"] = int(ymax * self.resolution[1])
        else:
            if self.resolution:
                label["x"] = x / self.resolution[0]
                label["y"] = y / self.resolution[1]
                label["w"] = w / self.resolution[0]
                label["h"] = h / self.resolution[1]
                label["xmin"] = int(xmin)
                label["ymin"] = int(ymin)
                label["xmax"] = int(xmax)
                label["ymax"] = int(ymax)
            else:
                print("[Data] ERROR, no resolution info")
                return
        # print(f"{len(seg[0]) = }\n{seg = }")    

        label["seg"] = seg
        # print(f"{seg_reshape = }")
        self.labels.append(label)
        # print(label)

    def __repr__(self):
        return f"image_name = {self.image_name},\n labels : {self.labels}"

        

        
