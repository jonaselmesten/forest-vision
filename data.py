import cv2
import json
import os

from detectron2.structures import BoxMode

class_id = {"stem-spruce": 1,
            "stem-birch": 2,
            "stem-pine": 3,
            "crown-spruce": 4,
            "crown-birch": 5,
            "crown-pine": 6}


def vgg_to_json(img_dir, file_name):
    annotation_counter = 0

    coco_data = {
        "info": {"description": "Swedish trees"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "stem", "id": 1, "name": "stem-birch"}
            #{"supercategory": "stem", "id": 1, "name": "stem-birch"},
            #{"supercategory": "stem", "id": 2, "name": "stem-pine"},
            #{"supercategory": "crown", "id": 3, "name": "crown-spruce"},
            #{"supercategory": "crown", "id": 4, "name": "crown-birch"},
            #{"supercategory": "crown", "id": 5, "name": "crown-pine"}
        ]
    }

    json_file = os.path.join(img_dir, file_name)

    with open(json_file) as f:
        file_annotation = json.load(f)

    for idx, v in enumerate(file_annotation.values()):
        record = {}

        try:
            filename = os.path.join(img_dir, v["filename"])
            height, width = cv2.imread(filename).shape[:2]
        except AttributeError as e:
            continue

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        coco_data["images"].append({
            "file_name": filename,
            "height": height,
            "width": width,
            "id": idx
        })

        data = v["regions"]

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
                "image_id": idx,
                "bbox": [min(px), min(py), max(px), max(py)],
                #"category_id": class_id[class_name],
                "category_id": 1,
                "id": annotation_counter
            }

            annotation_counter += 1

            coco_data["annotations"].append(annotation_obj)

    return coco_data

def vgg_merge(files):

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



