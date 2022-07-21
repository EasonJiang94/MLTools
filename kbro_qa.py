import os
from dataset import Dataset
from data import Data
import cv2
from eval_tool import EvalTool
    
if __name__ == "__main__":
    IOU_THRES = 0.3
    EvalTool.set_iou_thres(IOU_THRES)
    eval_tool = EvalTool()
    gt_image_dir = "/mnt/HD/tmp/project/Kbro/HD_000001-003732/tmp"
    
    ## set orig_KB_test3000_3 as gt ##
    yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/orig_KB_test3000_3/labels"
    yolov5_label_dir = "/mnt/HD/Workspace/yolov5_2/runs/detect/kbqa_test_/labels"
    eval_tool.add_yolo_dataset(gt_image_dir, yolov5_label_dir, gt=True)
    # yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/oil_ele_KB_test3000_3"
    # yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/oil_ele_KB_test3000_3/labels"
    # eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    # print(eval_tool)
    # eval_tool.run_benchmark(failed_dir="oil_ele_KB_test3000_3", draw=True)
    yolov5_image_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/kb_KB_test3000_3"
    yolov5_label_dir = "/mnt/HD/Workspace/yolov5_3/runs/detect/kb_KB_test3000_3/labels"
    eval_tool.add_yolo_dataset(yolov5_image_dir, yolov5_label_dir, pred=True)
    print(eval_tool)
    eval_tool.run_benchmark(failed_dir="kb_KB_test3000_3_vs_orig", draw=True)


    
    
