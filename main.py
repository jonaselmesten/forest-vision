import json
import logging
import os
import random
import time

import cv2
import numpy
from PIL.Image import fromarray
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode, Boxes
from detectron2.utils.logger import setup_logger, log_every_n
from detectron2.utils.visualizer import Visualizer, ColorMode
from matplotlib import pyplot as plt

from augmentation import image_augmentation
from config import cfg, metadata_train
from annotation import vgg_to_data_dict, vgg_val_split
from train import CustomTrainer, load_json_arr


def show_random_annotation():
    data_dicts = vgg_to_data_dict("stem/train")

    for d in random.sample(data_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_train, scale=0.5)
        out = visualizer.draw_dataset_dict(d)

        cv2.imshow("Annotations", out.get_image()[:, :, ::-1])
        cv2.waitKey()


def show_train_graph():
    experiment_metrics = load_json_arr(cfg.OUTPUT_DIR + '/metrics.json')

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title('Training val/loss')
    ax1.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    ax1.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])

    ax1.legend(['total_loss', 'validation_loss', 'loss_mask'], loc='upper left')

    ax1.set_title('Mask/BBox loss')
    ax2.plot(
        [x['iteration'] for x in experiment_metrics if 'loss_mask' in x],
        [x['loss_mask'] for x in experiment_metrics if 'loss_mask' in x])
    ax2.plot(
        [x['iteration'] for x in experiment_metrics if 'loss_box_reg' in x],
        [x['loss_box_reg'] for x in experiment_metrics if 'loss_box_reg' in x])

    ax2.legend(['loss_mask', 'loss_box'], loc='upper left')

    plt.show()


def train(show_graphs=True):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    show_train_graph()


def show_augmentation():
    dataset_dicts = vgg_to_data_dict("stem/train")

    for data_dict in random.sample(dataset_dicts, 100):

        data_dict_aug = image_augmentation(data_dict)
        instance = data_dict_aug["instances"]

        bboxes = instance.gt_boxes
        len_bboxes = len(bboxes)
        classes = instance.gt_classes
        polygon_masks = instance.gt_masks

        annotations = []

        # Gather annotation data.
        for i in range(len(data_dict["annotations"])):

            if i >= len_bboxes:
                break

            bbox = bboxes[i].tensor.tolist()[0]
            category_id = classes[0].tolist()
            polygon_mask = polygon_masks[i]
            segmentation = None

            for mask in polygon_mask:
                for xy in mask:
                    segmentation = [list(xy)]

            annotations.append({"bbox": bbox,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "segmentation": segmentation,
                                "category_id": category_id})

        data_dict["annotations"] = annotations
        image = data_dict_aug["image"]

        visualizer = Visualizer(image[:, :, ::-1], metadata=metadata_train, scale=0.8)
        out = visualizer.draw_dataset_dict(data_dict)

        cv2.imshow("Augmented image", out.get_image()[:, :, ::-1])
        cv2.waitKey()


def run_prediction(img, predictor):
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

    v = Visualizer(img[:, :, ::-1],
                   MetadataCatalog.get("stem_train"),
                   instance_mode=ColorMode(1))

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out = out.get_image()[:, :, ::-1]
    cv2.imshow(win_name, out)
    cv2.resizeWindow(win_name, 800, 600)
    cv2.waitKey()


def run_prediction_on_dir(img_dir):
    for file in os.listdir(img_dir):
        if file.split(".")[1] == "json":
            continue
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
        predictor = DefaultPredictor(cfg)
        run_prediction(img_dir + "/" + file, predictor)


show_augmentation()

# run_prediction_on_dir("stem/val")
# vgg_val_split("imgs", "stem/train", "stem/val", "imgs/data.json", 0.2)
