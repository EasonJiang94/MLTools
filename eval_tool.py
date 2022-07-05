import os
from dataset import Dataset

def make_if_inexist(script_dir: str, round=0) -> str:
    script_dir_path = f"{script_dir}_{round:02d}"
    if not os.path.exists(script_dir_path):
        os.makedirs(script_dir_path)
    else:
        make_if_inexist(script_dir, round+1)

class EvalTool(object):
    epsilon = 1E-6
    iou_thres = 0.5
    def __init__(self):
        self.gt = Dataset("GT")
        self.pred = Dataset("Pred")
        pass

    def add_yolo_dataset(self, img_dir, label_dir, gt=False, pred=False, attribute=None):
        if gt:
            self.gt.import_from_yolo(img_dir, label_dir, attribute=attribute)
        if pred:
            self.pred.import_from_yolo(img_dir, label_dir, attribute=attribute)
        

    def add_labelme_dataset(self, img_dir, label_dir, gt=False, pred=False, attribute=None):
        if gt:
            self.gt.import_from_labelme(img_dir, label_dir, attribute=attribute)
        if pred:
            self.pred.import_from_labelme(img_dir, label_dir, attribute=attribute)

    def _plot_result(self, failed_dir="FailedDir"):
        pass

    def _compare_bboxes_and_draw(self, gt_data, pred_data, failed_dir="FailedDir"):
        gt_labels = gt_data.labels
        pred_labels = pred_data.labels
        hit_pred = []
        for gt in gt_labels:
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
                if max_iou < iou:
                    max_iou = iou
                    max_pred_idx = pred_idx
            if max_pred_idx >= 0:
                pred_labels.pop(max_pred_idx)
            if max_iou < 1 and max_iou >= 0:
                print(f"{max_iou = }, {max_pred_idx = }")

        pass

    def run_benchmark(self, failed_dir="FailedDir"):
        print("start running benchmark")
        failed_dir = make_if_inexist(failed_dir)
        tp_total = 0
        fp_total = 0
        fn_total = 0
        tn_total = 0 # no label in image and no prediction
        unmatched_cnt = 0
        ### base on GT ###
        data_list = self.gt.data_dict.keys()
        for image_name in data_list:
            pred_data = self.pred.data_dict.get(image_name)
            gt_data   = self.gt.data_dict.get(image_name)
            if pred_data is None :
                print(f"Unmatch data : {image_name}")
            
            self._compare_bboxes_and_draw(gt_data, pred_data)
            

            # print(gt_data)
        
        recall = tp_total / (tp_total + fn_total + EvalTool.epsilon)
        precision = tp_total / (tp_total + fp_total + EvalTool.epsilon)
        accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + EvalTool.epsilon)
        print(f"Precision   : {precision*100:.2f}%")
        print(f"Recall rate : {recall*100:.2f}%")
        print(f"Accuracy    : {accuracy*100:.2f}%")
        # print(f"Precision : {precision*100:.2f}%")

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
    



if __name__ == "__main__":
    yolov5_image_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep001_"
    yolov5_label_dir = "/media/ubuntu/4f961193-2e0f-47b1-ade3-27999c0eabbf/detect/0601_sep001_/labels"

    eval_tool = EvalTool()
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    labelme_label_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep001_Done/json"
    labelme_image_dir = "/media/ubuntu/a689a8e0-c15c-406d-816b-6c1d619b9930/kbro_06/0601_sep001_Done/images"
    eval_tool.add_labelme_dataset(labelme_image_dir, labelme_label_dir, gt=True)

    
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, gt=True, attribute="auto_label")
    print(eval_tool)
    eval_tool.run_benchmark()