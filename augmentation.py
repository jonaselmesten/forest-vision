import os
import pathlib

import PIL
import torch
from PIL.Image import Image

import csv


def flip_img_csv(image_path, csv_path, save_folder):
    """
    Takes an image and its csv-bounding box data and flips it to
    create new images with matching csv-data.
    @param image_path: Path to image to be flipped.
    @param csv_path: Path to csv-file.
    @param save_folder: Folder to save the images and the csv-data.
    """

    org_img = Image.open(image_path)
    org_name = os.path.basename(image_path).split(sep=".")[0]
    width, height = org_img.size
    csv_data = []

    # Read in original csv data.
    with open(csv_path, "r", newline="\n", encoding="utf-8") as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            csv_data.append(row)

    for i in range(1, 4):

        file_path = os.path.join(save_folder, org_name + "_" + str(i))

        with open(file_path + ".csv", "w", newline="\n", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file, lineterminator='\n')

            for row in csv_data:

                x_min = int(row[1])
                y_min = int(row[2])
                x_max = int(row[3])
                y_max = int(row[4])
                bb_width = x_max - x_min
                bb_height = y_max - y_min

                # Flip the bounding box data.
                if i == 1:
                    x_min = width - x_min - bb_width
                    x_max = x_min + bb_width
                elif i == 2:
                    y_min = height - y_min - bb_height
                    y_max = y_min + bb_height
                    pass
                elif i == 3:
                    x_min = width - x_min - bb_width
                    x_max = x_min + bb_width
                    y_min = height - y_min - bb_height
                    y_max = y_min + bb_height
                elif i == 4:
                    csv_writer.writerow([os.path.basename(image_path), x_min, y_min, x_max, y_max, "Tree"])
                csv_writer.writerow([org_name + "_" + str(i) + ".JPG", x_min, y_min, x_max, y_max, "Tree"])

        if i == 1:
            flip_img = org_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            flip_img.save(file_path + ".jpg", "JPEG", quality=95)
        elif i == 2:
            flip_img = org_img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            flip_img.save(file_path + ".jpg", "JPEG", quality=95)
        elif i == 3:
            flip_img = org_img.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT)
            flip_img.save(file_path + ".jpg", "JPEG", quality=95)


from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


def custom_mapper(dataset_dict):

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    transform_list = [
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict
