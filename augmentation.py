import copy

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils


def image_augmentation(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    transform_list = [
        T.RandomCrop(crop_type="relative_range", crop_size=[0.9, 0.8]),
        T.RandomBrightness(0.9, 1.3),
        T.RandomContrast(0.8, 1.5),
        T.RandomSaturation(1.0, 1.4),
        T.RandomRotation(angle=[15, 0, 5, 6, 15], expand=False),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        #T.ResizeScale(1.0, 2.0, target_height=900, target_width=700)
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = image

    annotations = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    instances = utils.annotations_to_instances(annotations, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict
