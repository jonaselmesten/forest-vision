import copy

import detectron2.data.transforms as T
import torch
from detectron2.data import detection_utils as utils


def image_augmentation(dataset_dict):

    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="RGB")

    transform_list = [
        T.RandomBrightness(0.9, 1.2),
        T.RandomContrast(0.8, 1.4),
        T.RandomSaturation(0.1, 1.4),
        T.RandomRotation(angle=[1, 10], expand=False),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
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
