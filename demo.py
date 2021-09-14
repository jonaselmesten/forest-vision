import os
import time

import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

from model.config import cfg_instance, cfg_semantic
from model.predictor import SemanticPredictor, InstancePredictor
from visualize import CustomVisualizer


def run_instance_prediction_on_dir(img_dir):
    for file in os.listdir(img_dir):

        if file.split(".")[1] == "json":
            continue

        cfg_instance.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        predictor = SemanticPredictor(cfg_instance)

        run_instance_prediction(img_dir + "/" + file, predictor)


def run_instance_batch_prediction(img_dir, num_of_img=4, num_of_cycles=1):
    predictor = SemanticPredictor(cfg_instance)

    img_list = []
    output_list = []
    start = time.time()

    for cycle in range(num_of_cycles):
        for i, file in enumerate(os.listdir(img_dir)):
            if file.split(".")[1] == "json":
                continue

            img_list.append(cv2.imread(os.path.join(img_dir, file)))

            if i == num_of_img:
                break

        output = predictor.batch_process(img_list)
        output_list.extend(output)

        if cycle == num_of_cycles:
            break

        for img, result in zip(img_list, output_list):
            print(img)
            print(output)

            v = Visualizer(img[:, :, ::-1],
                           instance_mode=ColorMode(1))

            out = v.draw_instance_predictions(result["instances"].to("cpu"))
            out = out.get_image()[:, :, ::-1]
            cv2.imshow("win_name", out)
            cv2.resizeWindow("win_name", 800, 600)
            cv2.waitKey()

        img_list.clear()
        output_list.clear()


def run_panoptic_instance_prediction_on_dir(img_dir):
    for file in os.listdir(img_dir):

        if file.split(".")[1] == "json":
            continue

        run_panoptic_instance_prediction(os.path.join(img_dir, file))


def run_panoptic_instance_prediction(img):
    img = cv2.imread(img)
    win_name = "Prediction"

    instance_predictor = InstancePredictor(cfg_instance)
    semantic_predictor = SemanticPredictor(cfg_semantic)

    outputs = instance_predictor(img)
    img_seg, seg_info = semantic_predictor(img)["panoptic_seg"]

    v = CustomVisualizer(img[:, :, ::-1],
                         metadata=MetadataCatalog.get("stem_train"),
                         metadata_semantic=MetadataCatalog.get(cfg_semantic.DATASETS.TRAIN[0]),
                         instance_mode=ColorMode(1))

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out = v.draw_panoptic_seg(img_seg.to("cpu"), seg_info)

    out = out.get_image()[:, :, ::-1]
    cv2.imshow(win_name, out)
    cv2.resizeWindow(win_name, 800, 600)
    cv2.waitKey()


def run_semantic_instance_prediction(img, threshold=0.8, resize=100):

    cfg_instance.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    img = cv2.imread(img)

    if resize < 100:
        scale_percent = resize
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    win_name = "Prediction"

    instance_predictor = InstancePredictor(cfg_instance)
    semantic_predictor = SemanticPredictor(cfg_semantic)

    outputs = instance_predictor(img)
    img_seg = semantic_predictor(img)

    v = CustomVisualizer(img[:, :, ::-1],
                         metadata=MetadataCatalog.get("stem_train"),
                         metadata_semantic=MetadataCatalog.get(cfg_semantic.DATASETS.TRAIN[0]),
                         instance_mode=ColorMode(1))

    out = v.draw_sem_seg(img_seg["sem_seg"].argmax(dim=0).to("cpu"))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    out = out.get_image()[:, :, ::-1]
    cv2.imshow(win_name, out)
    cv2.resizeWindow(win_name, 800, 600)
    cv2.waitKey()


def run_instance_prediction(img, predictor):
    cfg_instance.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    img = cv2.imread(img)
    win_name = "Prediction"

    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1],
                   instance_mode=ColorMode(1))

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out = out.get_image()[:, :, ::-1]
    cv2.imshow(win_name, out)
    cv2.resizeWindow(win_name, 800, 600)
    cv2.waitKey()


def run_semantic_prediction_on_dir(img_dir):
    for file in os.listdir(img_dir):

        if file.split(".")[1] == "json":
            continue

        predictor = SemanticPredictor(cfg_semantic)

        run_semantic_prediction(img_dir + "/" + file, predictor)


def run_semantic_instance_prediction_on_dir(img_dir):
    for file in os.listdir(img_dir):

        if file.split(".")[1] == "json":
            continue

        run_semantic_instance_prediction(img_dir + "/" + file)


def run_semantic_prediction(img, predictor):
    img = cv2.imread(img)
    win_name = "Prediction"

    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1],
                   metadata=MetadataCatalog.get(cfg_semantic.DATASETS.TRAIN[0]),
                   instance_mode=ColorMode(1))

    out = v.draw_sem_seg(outputs["sem_seg"].argmax(dim=0).to("cpu"))

    out = out.get_image()[:, :, ::-1]
    cv2.imshow(win_name, out)
    cv2.resizeWindow(win_name, 800, 600)
    cv2.waitKey()
