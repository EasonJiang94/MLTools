from copy import deepcopy as dcp

class Data(object):
    idx = 0
    label_template = {
        "class_name" : "human",
        "attribute" : [],
        "x" : 0.0,
        "y" : 0.0,
        "w" : 0.0,
        "h" : 0.0,
        "xmin" : 0.0,        
        "ymin" : 0.0,
        "xmax" : 0.0,
        "ymax" : 0.0,        
        "c" : 1.0 ## means confidence score, conf of gt always be 1.0
    }
    def __init__(self, image_name="Unknown"):
        self.image_name = image_name
        self.image_path = "Unknown"
        self.labels = []
        self.ir = False
        self.cam_name = "Unknown"
        self.resolution = None # w, h
        
    @property
    def label_num(self) ->int:
        return len(self.labels)

    def set_resolution(self,w:int,h:int):
        self.resolution = (w,h)

    def set_resolution(self,resolution:tuple):
        assert len(resolution)==2, 'size of resolution must be 2'
        assert isinstance(resolution[0], int), 'element of resolution must be int'
        assert isinstance(resolution[1], int), 'element of resolution must be int'
        self.resolution = resolution

    def add_by_xyxy(self, xyxy:list, \
                    ratio=True, \
                    conf=1.0, \
                    class_name=None, \
                    attribute=None):
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
        if attribute:
            label["attribute"] = attribute

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
                    attribute=None):
        
        

        label = dcp(Data.label_template)
        if class_name:
            label["class_name"] = class_name
        if attribute:
            label["attribute"] = attribute
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

    def __repr__(self):
        return f"image_name = {self.image_name},\n labels : {self.labels}"

        

        