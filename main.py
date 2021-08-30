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
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from augmentation import image_augmentation
from config import cfg
from data import vgg_to_data_dict, vgg_val_split

setup_logger()

for mode in ["train", "val"]:
    DatasetCatalog.register("stem_" + mode, lambda d=mode: vgg_to_data_dict("stem/train", "merged.json"))
    MetadataCatalog.get("stem_" + mode).set(
        thing_classes=["spruce", "birch", "pine", "spruce-crown", "birch-crown", "pine-crown"])

metadata_train = MetadataCatalog.get("stem_train")


def show_random_annotation():
    data_dicts = vgg_to_data_dict("stem/train", "merged.json")

    for d in random.sample(data_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_train, scale=0.5)
        out = visualizer.draw_dataset_dict(d)

        cv2.imshow("Annotations", out.get_image()[:, :, ::-1])
        cv2.waitKey()


def train():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = CustomTrainer(cfg)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def show_augmentation():
    dataset_dicts = vgg_to_data_dict("stem/train", "merged.json")

    for data_dict in random.sample(dataset_dicts, 3):

        data_dict_aug = image_augmentation(data_dict)
        instance = data_dict_aug["instances"]

        bboxes = instance.gt_boxes
        classes = instance.gt_classes
        polygon_masks = instance.gt_masks

        annotations = []

        for i in range(len(data_dict["annotations"])):

            bbox = bboxes[i].tensor.tolist()[0]
            category_id = classes[0].tolist()
            polygon_mask = polygon_masks[i]
            segmentation = None
            print(category_id)

            for mask in polygon_mask:
                for xy in mask:
                    segmentation = [list(xy)]

            annotations.append({"bbox": bbox,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "segmentation": segmentation,
                                "category_id": category_id})

        data_dict["annotations"] = annotations

        tensor = data_dict_aug["image"][0]
        pil_image = fromarray(tensor.numpy())
        img = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_BGR2RGB)

        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_train, scale=0.5)
        out = visualizer.draw_dataset_dict(data_dict)

        cv2.imshow("Augmented image", out.get_image()[:, :, ::-1])
        cv2.waitKey()


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


def run_all_val_pred():
    for file in os.listdir("stem/val"):
        print("SHOWING:", file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
        predictor = DefaultPredictor(cfg)
        make_prediction("stem/val/" + file, predictor)

vgg_val_split("imgs", "stem/train", "stem/val", "merged.json", 0.2)
