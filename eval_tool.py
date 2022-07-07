import os
from dataset import Dataset
from data import Data
import cv2

def make_if_inexist_recursive(script_dir: str, round=0) -> str:
    script_dir_path = f"{script_dir}_{round:02d}"
    if not os.path.exists(script_dir_path):
        os.makedirs(script_dir_path)
        return script_dir_path
    else:
        return make_if_inexist_recursive(script_dir,round+1)
        
def make_if_inexist(script_dir: str) -> str:
    script_dir_path = f"{script_dir}"
    if not os.path.exists(script_dir_path):
        os.makedirs(script_dir_path)
        
    

class EvalTool(object):
    epsilon = 1E-6
    iou_thres = 0.5
    GT_COLOR = (30, 30, 220)
    HIT_COLOR = (100, 255, 100)
    FP_COLOR = (120, 120, 235)
    LABEL_TEXT_COLOR = (245,245,225)
    def __init__(self):
        self.gt = Dataset("GT")
        self.pred = Dataset("Pred")
        self.gt_black_list = ["no_human_but_detected"]
        pass

    def add_yolo_dataset(self, img_dir, label_dir, gt=False, pred=False, attribute=None):
        if gt:
            self.gt.import_from_yolo(img_dir, label_dir, attribute=attribute, set_ir_flag=True, class_filter=[0])
        if pred:
            self.pred.import_from_yolo(img_dir, label_dir, attribute=attribute, class_filter=["0"])
        

    def add_labelme_dataset(self, img_dir, label_dir, gt=False, pred=False, attribute=None, blacklist=[]):
        if gt:
            self.gt.import_from_labelme(img_dir, label_dir, attribute=attribute, set_ir_flag=True, blacklist=blacklist)
        if pred:
            self.pred.import_from_labelme(img_dir, label_dir, attribute=attribute, set_ir_flag=False)

    def _init_statics_by_class(self, class_name):
        self.tp_by_class[class_name] = 0
        self.fp_by_class[class_name] = 0
        self.fn_by_class[class_name] = 0
        self.tn_by_class[class_name] = 0

    def _draw_and_output(self, gt_data, paired_pred_list, unpaired_pred_list, output_dir, by_class=True):
        gt_labels = gt_data.labels
        image = cv2.imread(gt_data.image_path)
        class_list = ["all"]
        for gt_label in gt_labels:
            xmin = gt_label["xmin"]
            ymin = gt_label["ymin"]
            xmax = gt_label["xmax"]
            ymax = gt_label["ymax"]
            class_name = gt_label["class_name"]
            if (class_name not in class_list) and by_class:
                class_list.append(class_name)
            t_size = cv2.getTextSize(class_name, 0, fontScale=1, thickness=2)[0]
            cv2.rectangle(image, (xmin, ymin), (xmin+t_size[0], ymin - t_size[1]-3), EvalTool.GT_COLOR, -1, cv2.LINE_AA)
            cv2.putText(image, class_name, (xmin, ymin - 2), 0, 1, EvalTool.LABEL_TEXT_COLOR, thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), EvalTool.GT_COLOR, thickness=2)

        for paired_pred in paired_pred_list:
            xmin = paired_pred["xmin"]
            ymin = paired_pred["ymin"]
            xmax = paired_pred["xmax"]
            ymax = paired_pred["ymax"]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), EvalTool.HIT_COLOR, thickness=2)

        for unpaired_pred in unpaired_pred_list:
            xmin = unpaired_pred["xmin"]
            ymin = unpaired_pred["ymin"]
            xmax = unpaired_pred["xmax"]
            ymax = unpaired_pred["ymax"]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), EvalTool.FP_COLOR, thickness=2)
        
        for class_name in class_list:
            output_dir_path = os.path.join(output_dir, class_name)
            output_img_path = os.path.join(output_dir_path, gt_data.image_name)
            make_if_inexist(output_dir_path)
            cv2.imwrite(output_img_path, image)



    def _compare_bboxes_and_draw(self, image_name:str, gt_data:Data, pred_data:Data, failed_dir="FailedDir", draw=False):
        gt_labels = gt_data.labels
        pred_labels = pred_data.labels
        tp, fp, tn, fn, = 0, 0, 0, 0
        hit_pred = []
        paired_gt_list = []
        paired_pred_list = []
        unpaired_gt_list = []
        unpaired_pred_list = []
        for gt_idx, gt in enumerate(gt_labels):
            gt_xmin = gt["xmin"]
            gt_ymin = gt["ymin"]
            gt_xmax = gt["xmax"]
            gt_ymax = gt["ymax"]
            max_iou = -1
            max_pred_idx = -1
            # print("----")
            for pred_idx, pred in enumerate(pred_labels):
                pred_xmin = pred["xmin"]
                pred_ymin = pred["ymin"]
                pred_xmax = pred["xmax"]
                pred_ymax = pred["ymax"]
                iou = self._get_iou((gt_xmin,gt_ymin,gt_xmax,gt_ymax), (pred_xmin,pred_ymin,pred_xmax,pred_ymax))
                # print(iou)
                if max_iou < iou and iou > EvalTool.iou_thres:
                    max_iou = iou
                    max_pred_idx = pred_idx
            if max_pred_idx >= 0:
                paired_pred_list.append(pred_labels.pop(max_pred_idx))
                paired_gt_list.append(gt)
            else : 
                unpaired_gt_list.append(gt)
        
        unpaired_pred_list = pred_labels
        tp = len(paired_gt_list)
        fp = len(unpaired_pred_list)
        fn = len(unpaired_gt_list)
        if tp == 0 and fp == 0 and tn == 0:
            tn =1
        
        for paired_gt in paired_gt_list:
            class_name = paired_gt["class_name"]
            if class_name in self.tp_by_class:
                pass
            else:
                self._init_statics_by_class(class_name)
            self.tp_by_class[class_name] += 1

        for unpaired_pred in unpaired_pred_list:
            class_name = unpaired_pred["class_name"]
            if class_name in self.fp_by_class:
                pass
            else:
                self._init_statics_by_class(class_name)
            self.fp_by_class[class_name] += 1

        for unpaired_gt in unpaired_gt_list:
            class_name = unpaired_gt["class_name"]
            if class_name in self.fn_by_class:
                pass
            else:
                self._init_statics_by_class(class_name)
            self.fn_by_class[class_name] += 1
        
        
        if draw:
            self._draw_and_output(gt_data, paired_pred_list, unpaired_pred_list, failed_dir)

        return tp, fp , tn, fn

    def run_benchmark(self, failed_dir="FailedDir", draw=False):
        print("start running benchmark")
        failed_dir = make_if_inexist_recursive(failed_dir)
        tp_total = 0
        fp_total = 0
        fn_total = 0
        tn_total = 0 # no label in image and no prediction
        unmatched_cnt = 0
        self.tp_by_class = {}
        self.fp_by_class = {}
        self.fn_by_class = {}
        self.tn_by_class = {}
        ## base on GT ###
        image_name_list = self.gt.data_dict.keys()
        for iter_idx, image_name in enumerate(image_name_list):
            print(f"run_benchmark : {iter_idx / len(image_name_list)*100:2.2f}%", end='\r')

            pred_data = self.pred.data_dict.get(image_name)
            gt_data   = self.gt.data_dict.get(image_name)
            if pred_data is None :
                print(f"Unmatch data : {image_name}")
            else:
                tp, fp, tn, fn = self._compare_bboxes_and_draw(image_name, gt_data, pred_data, failed_dir=failed_dir, draw=draw)
                tp_total += tp
                fp_total += fp
                fn_total += fn
                tn_total += tn
        print(f"run_benchmark : 100.00%", end='\n')
            

        ### base on pred ###
        # data_list = self.pred.data_dict.keys()
        # for image_name in data_list:
        #     pred_data = self.pred.data_dict.get(image_name)
        #     gt_data   = self.gt.data_dict.get(image_name)
        #     if gt_data is None :
        #         print(f"Unmatch data : {image_name}")
        #     else:
        #         self._compare_bboxes_and_draw(image_name, gt_data, pred_data)
        
        
        recall = tp_total / (tp_total + fn_total + EvalTool.epsilon)
        precision = tp_total / (tp_total + fp_total + EvalTool.epsilon)
        accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + EvalTool.epsilon)
        print(f"Precision   : {precision*100:3.2f}%")
        print(f"Recall rate : {recall*100:3.2f}%")
        print(f"Accuracy    : {accuracy*100:3.2f}%")
        print(f"=====[show by case]======")
        for class_name in self.tp_by_class.keys():
            print(f"Class Name  : <{class_name}>")
            precision = self.tp_by_class[class_name] / ((self.tp_by_class[class_name] + self.fp_by_class[class_name] + EvalTool.epsilon))
            recall = self.tp_by_class[class_name] / (self.tp_by_class[class_name] + self.fn_by_class[class_name] + EvalTool.epsilon)
            print(f"Precision   : {precision*100:3.2f}%,\t({self.tp_by_class[class_name]:3d}/{(self.tp_by_class[class_name] + self.fp_by_class[class_name]):3d})")
            print(f"Recall rate : {recall*100:3.2f}%,\t({self.tp_by_class[class_name]:3d}/{(self.tp_by_class[class_name] + self.fn_by_class[class_name]):3d})")
            print("")
        # print(f"{self.tp_by_class = }")
        # print(f"{self.fp_by_class = }")
        # print(f"{self.fn_by_class = }")
        # print(f"{self.tn_by_class = }")
        print(f"=========================")

    def _get_iou(self, xyxy1, xyxy2):
        boxA = [0,0,0,0]
        boxB = [0,0,0,0]
        
        # print(f"{xyxy2[0] = }")
        boxB[0] = int(xyxy2[0])
        boxB[1] = int(xyxy2[1])
        boxB[2] = int(xyxy2[2])
        boxB[3] = int(xyxy2[3])
        boxA[0] = int(xyxy1[0])
        boxA[1] = int(xyxy1[1])
        boxA[2] = int(xyxy1[2])
        boxA[3] = int(xyxy1[3])

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
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

    def __repr__(self):
        return f"GT Dataset : \n{self.gt}\nPred Dataset : \n{self.pred}\n"

    @classmethod
    def set_iou_thres(cls, iou_thres):
        assert isinstance(iou_thres, float)
        assert iou_thres>=0 or iou_thres<=1, "iou_thres should >= 0 and <= 1"
        cls.iou_thres = iou_thres
    



if __name__ == "__main__":
    IOU_THRES = 0.3
    EvalTool.set_iou_thres(IOU_THRES)
    eval_tool = EvalTool()
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep001_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep001_/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep002_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep002_/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep003_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep003_/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep004_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep004_/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep005_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep005_/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep006_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep006_/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep007_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep007_/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep008_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep008_/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep009_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep009_/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep010_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep010_/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)

    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_001_"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_001_/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_002_"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_002_/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_003_"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_003_/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_004_"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_004_/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_005_"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_005_/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_006_"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_006_/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_007_"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_007_/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_008_"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_008_/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_009_"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_009_/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_010_"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/sep_010_/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)





    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep001_/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_0601/data2/seps/001"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True, blacklist=["no_human_but_detected"])
    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep002_/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_0601/data2/seps/002"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True, blacklist=["no_human_but_detected"])
    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep003_/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_0601/data2/seps/003"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True, blacklist=["no_human_but_detected"])
    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep004_/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_0601/data2/seps/004"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True, blacklist=["no_human_but_detected"])
    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep005_/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_0601/data2/seps/005"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True, blacklist=["no_human_but_detected"])
    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep006_/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_0601/data2/seps/006"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True, blacklist=["no_human_but_detected"])
    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep007_/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_0601/data2/seps/007"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True, blacklist=["no_human_but_detected"])
    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep008_/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_0601/data2/seps/008"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True, blacklist=["no_human_but_detected"])
    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep009_/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_0601/data2/seps/009"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True, blacklist=["no_human_but_detected"])
    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep010_/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_0601/data2/seps/010"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True, blacklist=["no_human_but_detected"])
    
    print(eval_tool)
    eval_tool.run_benchmark(draw=False)
