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
from detectron2.utils.visualizer import Visualizer, ColorMode
from matplotlib import pyplot as plt

from augmentation import image_augmentation
from config import cfg
from data import vgg_to_data_dict, vgg_val_split
from train import CustomTrainer, load_json_arr

setup_logger()

for mode in ["train", "val"]:
    DatasetCatalog.register("stem_" + mode, lambda d=mode: vgg_to_data_dict("stem/" + mode))
    MetadataCatalog.get("stem_" + mode).set(
        thing_colors=[(250, 0, 0), (25, 255, 25), (34, 0, 204), (0, 0, 255), (0, 0, 255), (0, 0, 255)],
        thing_classes=["spruce", "birch", "pine", "spruce-crown", "birch-crown", "pine-crown"])

metadata_train = MetadataCatalog.get("stem_train")


def show_random_annotation():
    data_dicts = vgg_to_data_dict("stem/train")

    for d in random.sample(data_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_train, scale=0.5)
        out = visualizer.draw_dataset_dict(d)

        cv2.imshow("Annotations", out.get_image()[:, :, ::-1])
        cv2.waitKey()


def train():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

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


def show_augmentation():
    dataset_dicts = vgg_to_data_dict("stem/train")

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
        img = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)

        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_train, scale=0.5)
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
                   scale=0.8,
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


