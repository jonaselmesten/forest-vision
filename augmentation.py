import copy
import random

import cv2
import detectron2.data.transforms as T
import torch
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

from annotation import vgg_to_data_dict
from model.config import metadata_train


def image_augmentation(dataset_dict):
    """
    Method to create image augmentation for training.
    :param dataset_dict: Data dict of one image.
    :return: Augmented data dict.
    """
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    transform_list = [
        T.RandomCrop(crop_type="relative_range", crop_size=[0.9, 0.8]),
        T.RandomBrightness(0.9, 1.3),
        T.RandomContrast(0.8, 1.5),
        T.RandomSaturation(1.0, 1.4),
        T.RandomRotation(angle=[15, 0, 5, 6, 15], expand=False),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        # T.ResizeScale(1.0, 2.0, target_height=900, target_width=700)
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annotations = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    instances = utils.annotations_to_instances(annotations, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


def show_augmentation(samples=100):
    """
    Visualizes how the data augmentations look.
    Press enter to move to the next image.
    """
    dataset_dicts = vgg_to_data_dict("stem/train")

    for data_dict in random.sample(dataset_dicts, samples):

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
        image = image.numpy().transpose(1, 2, 0)

        visualizer = Visualizer(image[:, :, ::-1], metadata=metadata_train, scale=0.8)
        out = visualizer.draw_dataset_dict(data_dict)

        cv2.imshow("Augmented image", out.get_image()[:, :, ::-1])
        cv2.waitKey()
