import shutil
from os.path import exists
import random

import cv2
import json
import os

import numpy as np
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

from model import metadata_train

categories = [
            {"supercategory": "stem", "id": 0, "name": "stem-birch"},
            {"supercategory": "stem", "id": 1, "name": "stem-birch"},
            {"supercategory": "stem", "id": 2, "name": "stem-pine"},
            {"supercategory": "crown", "id": 3, "name": "crown-spruce"},
            {"supercategory": "crown", "id": 4, "name": "crown-birch"},
            {"supercategory": "crown", "id": 5, "name": "crown-pine"} ]

class_id = {"stem-spruce": 0,
            "stem-birch": 1,
            "stem-pine": 2,
            "crown-spruce": 3,
            "crown-birch": 4,
            "crown-pine": 5}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def vgg_to_data_dict(img_dir):
    """
    Turns a vgg-file into detectron data dicts.
    :param img_dir: Dir of images and the data.json file.
    :return: Data dicts.
    """
    with open(os.path.join(img_dir, "data.json")) as f:
        annotations = json.load(f)

    dataset_dicts = []

    for img_id, values in enumerate(annotations.values()):

        data_dict = {}

        if type(values) is list:
            continue

        if not exists(img_dir + "/" + values["filename"]):
            continue

        file_name = os.path.join(img_dir, values["filename"])
        height, width = cv2.imread(file_name).shape[:2]

        data_dict["file_name"] = file_name
        data_dict["image_id"] = img_id
        data_dict["height"] = height
        data_dict["width"] = width

        regions = values["regions"]
        objects = []

        for region in regions:

            class_name = region["region_attributes"]["class"]

            if type(class_name) is dict:
                for key, _ in class_name.items():
                    class_name = key

            shape_attributes = region["shape_attributes"]

            px = shape_attributes["all_points_x"]
            py = shape_attributes["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": class_id[class_name]
            }

            objects.append(obj)

        data_dict["annotations"] = objects
        dataset_dicts.append(data_dict)

    return dataset_dicts


def vgg_to_coco(img_dir, file_name):
    """
    Turns a vgg-file into coco-format.
    :param img_dir: Image die.
    :param file_name: File name of vgg-file.
    :return: coco-format json.
    """

    annotation_counter = 0

    coco_data = {
        "info": {"description": "Swedish trees"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    json_file = os.path.join(img_dir, file_name)

    with open(json_file) as f:
        file_annotation = json.load(f)

    for img_id, values in enumerate(file_annotation.values()):
        record = {}

        try:
            filename = os.path.join(img_dir, values["filename"])
            height, width = cv2.imread(filename).shape[:2]
        except AttributeError as e:
            continue

        record["file_name"] = filename
        record["image_id"] = img_id
        record["height"] = height
        record["width"] = width

        coco_data["images"].append({
            "file_name": filename,
            "height": height,
            "width": width,
            "id": img_id
        })

        data = values["regions"]

        if len(data) == 0:
            continue

        for annotation in data:
            class_name = annotation["region_attributes"]["class"]
            annotation = annotation["shape_attributes"]
            px = annotation["all_points_x"]
            py = annotation["all_points_y"]

            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            if type(class_name) is dict:
                for key, _ in class_name.items():
                    class_name = key

            annotation_obj = {
                "segmentation": [poly],
                "iscrowd": 0,
                "bbox_mode": BoxMode.XYXY_ABS,
                "image_id": img_id,
                "bbox": [min(px), min(py), max(px), max(py)],
                "category_id": class_id[class_name],
                "id": annotation_counter
            }

            annotation_counter += 1

            coco_data["annotations"].append(annotation_obj)

    return coco_data


def merge_coco_files(files):
    """
    Merges two or more coco-annotation-files into a single one.
    :param files: List of coco-files.
    """
    data = {}
    annotations = []
    images = []

    # Collect all image & annotation data.
    for file in files:
        with open(file) as f:
            annotation = json.load(f)
            for key, value in annotation.items():
                if key == "images":
                    images.extend(annotation[key])
                elif key == "annotations":
                    annotations.extend(annotation[key])
                else:
                    data[key] = value

    # Generate unique id.
    for index, image in enumerate(images):
        image["id"] = index
    for index, annotation in enumerate(annotations):
        annotation["id"] = index

    # Merge.
    data["images"] = images
    data["annotations"] = annotations

    with open("merged.json", "w") as out_file:
        out_file.write(json.dumps(data, indent=4))


def vgg_val_split(image_dir, train_dir, val_dir, json_file, val_percent):
    """
    Takes a folder with images and its annotation data and splits into training/val.
    Copies all the images and creates json-file in both train/val.
    :param image_dir: Folder with all images to make the split from.
    :param train_dir: Train dir.
    :param val_dir: Val dir.
    :param json_file: File with all annotation to the img dir.
    :param val_percent: How much of the images that will end up in val. 0.2 gives 20%.
    """

    with open(json_file) as in_file:

        vgg_dict = json.load(in_file)

        total = len(vgg_dict)
        val_count = round(total * val_percent)
        print("Total images:", total)
        print("Val images:", val_count)

        val_dict = {}
        key_list = [key for key in vgg_dict]

        for img in random.sample(key_list, val_count):
            val_dict[img] = vgg_dict[img]
            vgg_dict.pop(img)

        # Copy all files.
        for img in val_dict.keys():
            file_name = val_dict[img]["filename"]
            shutil.copy(os.path.join(image_dir, file_name), val_dir)
        for img in vgg_dict.keys():
            file_name = vgg_dict[img]["filename"]
            shutil.copy(os.path.join(image_dir, file_name), train_dir)

        # Create json-data split.
        with open(os.path.join(train_dir, "data.json"), "w") as out_file:
            out_file.write(json.dumps(vgg_dict, indent=4))
        with open(os.path.join(val_dir, "data.json"), "w") as out_file:
            out_file.write(json.dumps(val_dict, indent=4))


def show_random_instance_annotation():
    data_dicts = vgg_to_data_dict("stem/train")

    for data in random.sample(data_dicts, 3):
        img = cv2.imread(data["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_train, scale=0.5)
        out = visualizer.draw_dataset_dict(data)

        cv2.imshow("Annotations", out.get_image()[:, :, ::-1])
        cv2.waitKey()