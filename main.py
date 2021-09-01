import os
import random

import cv2
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt

from annotation import vgg_to_data_dict
from augmentation import image_augmentation
from config import cfg, metadata_train
from predict import run_batch_prediction
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





show_augmentation()
#run_batch_prediction("stem/train", num_of_img=4, num_of_cycles=25)

# run_prediction_on_dir("stem/val")
# vgg_val_split("imgs", "stem/train", "stem/val", "imgs/data.json", 0.2)
