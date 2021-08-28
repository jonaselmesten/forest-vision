import json
import os
import random
import time
from json import loads
# Some basic setup:
# Setup detectron2 logger
from os.path import exists

import detectron2
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from config import cfg
from data import vgg_to_json

setup_logger()

class_id = {"stem-spruce": 0,
            "stem-birch": 1,
            "stem-pine": 2,
            "crown-spruce": 4,
            "crown-birch": 5,
            "crown-pine": 6}


def get_balloon_dicts(img_dir):
    #json_file = os.path.join(img_dir, "birch.json")
    with open("merged.json") as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        if type(v) is list:
            print(v)
            print("LIST:", idx)
            continue

        if not exists(img_dir + "/" + v["filename"]):
            print(v)
            print("DONT EXIST:", idx)
            continue

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []

        for anno in annos:

            #assert not anno["region_attributes"]

            class_name = anno["region_attributes"]["class"]
            if type(class_name) is dict:
                for key, _ in class_name.items():
                    class_name = key

            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": class_id[class_name]
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


for d in ["train", "val"]:
    DatasetCatalog.register("stem_" + d, lambda d=d: get_balloon_dicts("stem/train"))
    MetadataCatalog.get("stem_" + d).set(thing_classes=["spruce", "birch", "pine", "no", "no1", "no2"])
balloon_metadata = MetadataCatalog.get("stem_train")


def show_annotation():
    json_data = get_balloon_dicts("stem/train")
    annotations = []

    for image_data in random.sample(json_data["images"], 30):
        img = cv2.imread(image_data["file_name"])

        for seg_obj in json_data["annotations"]:
            if seg_obj["image_id"] == image_data["id"]:
                annotations.append(seg_obj)

        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.65)
        out = visualizer.draw_dataset_dict({"annotations": annotations})
        cv2.imshow("win", out.get_image()[:, :, ::-1])
        cv2.waitKey()
        annotations.clear()

def train():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
def make_prediction(img, predictor):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

    img = cv2.imread(img)
    win_name = "Prediction"

    start_time = time.time()
    outputs = predictor(img)
    print("--- %s seconds ---" % (time.time() - start_time))

    classes = outputs["instances"].pred_classes
    print(classes)
    print("Classes:", len(classes))

    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out = out.get_image()[:, :, ::-1]
    cv2.imshow(win_name, out)
    cv2.resizeWindow(win_name, 800, 600)
    cv2.waitKey()


def show_vallala():
    dataset_dicts = get_balloon_dicts("stem/train")
    for d in random.sample(dataset_dicts, 30):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("win", out.get_image()[:, :, ::-1])
        cv2.waitKey()

def run_all_val_pred():
    for file in os.listdir("stem/val"):
        print("SHOWING:", file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
        predictor = DefaultPredictor(cfg)
        make_prediction("stem/val/" + file, predictor)

train()
run_all_val_pred()