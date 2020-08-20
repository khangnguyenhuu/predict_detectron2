import os
import cv2
import json
import random
import itertools
import numpy as np

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, evaluator
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

import json

def predict (path_weigths, path_config, confidence_threshold, num_of_class, path_img):
  cfg = get_cfg()
  cfg.merge_from_file(path_config)
  cfg.MODEL.WEIGHTS = path_weigths

  #cfg.MODEL.WEIGHTS = "mask_rcnn_R_50_FPN_3x_model/model_final.pth"
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8   
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class 
  predictor = DefaultPredictor(cfg)
  im = cv2.imread(path_img)
  outputs = predictor(im)
  
  return outputs

#Đầu vào detect = output của hàm predict, frame = original image của mình, classs = tên class để visualize
def visualize (detect, frame, classs):
  boxes = detect['instances'].pred_boxes
  scores = detect['instances'].scores
  classes = detect['instances'].pred_classes
  for i in range (len(classes)):
    if (classes[i] == 2 or classes[i] == 3 or classes[i] == 4 or classes[i] == 6 or classes[i] == 7 or classes[i] == 8):
      if (scores[i] > 0.5):
        for j in boxes[i]:
          start = (int (j[0]), int (j[1]))
          end = (int (j[2]), int (j[3]))
        color = int (classes[i])
        cv2.rectangle(frame, start, end, (random.randint(0,255),random.randint(0,255),255), 1)
        cv2.putText(frame, str (classs[color]),start, cv2.FONT_HERSHEY_PLAIN, 1, (random.randint(0,255),random.randint(0,255),255), 2)
  return frame

  def main:
    path_weigth = 
    path_config =
    confidences_threshold = 
    num_of_class = 
    path_img = 
    classes = ['di_bo','xe_dap','xe_may','xe_hang_rong','xe_ba_gac','xe_taxi','xe_hoi','xe_ban_tai','xe_cuu_thuong','xe_khach','xe_buyt','xe_tai','xe_container','xe_cuu_hoa']
    _frame = cv2.imread(path_img)
    outputs = predict(path_weigth, path_config, confidences_threshold, num_of_class, path_img)
    frame = visualize (outputs, _frame, classes )
    cv2_imshow(frame)
