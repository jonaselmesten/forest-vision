import os
import time

import cv2
import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer, ColorMode

from config import cfg_instance


class InstancePredictor:

    def __init__(self, cfg):

        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
        """

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258

            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            return predictions

    def batch_process(self, images):

        input_list = []

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258

            for img in images:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    img = img[:, :, ::-1]

                height, width = img.shape[:2]
                img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

                inputs = {"image": img, "height": height, "width": width}
                input_list.append(inputs)

            predictions = self.model(input_list)[0]

        return predictions


class SemanticPredictor:

    def __init__(self, cfg):

        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
        """

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258

            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            return predictions


def run_instance_prediction(img, predictor):
    cfg_instance.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg_instance.MODEL.WEIGHTS = os.path.join(cfg_instance.OUTPUT_DIR,
                                              "model_final.pth")  # path to the model we just trained

    img = cv2.imread(img)
    win_name = "Prediction"

    outputs = predictor(img)

    # for field in outputs["instances"].get_fields():
    #    print(outputs["instances"].get(field))

    # outputs["instances"].remove("pred_boxes")
    # outputs["instances"].remove("scores")

    # pred_boxes
    # scores
    # pred_classes
    # pred_masks

    v = Visualizer(img[:, :, ::-1],
                   instance_mode=ColorMode(1))

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out = out.get_image()[:, :, ::-1]
    cv2.imshow(win_name, out)
    cv2.resizeWindow(win_name, 800, 600)
    cv2.waitKey()


def run_instance_prediction_on_dir(img_dir):
    for file in os.listdir(img_dir):

        if file.split(".")[1] == "json":
            continue

        cfg_instance.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
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
        img_list.clear()

        if cycle == num_of_cycles:
            break

    stop = time.time()
    img_count = num_of_cycles * num_of_img

    print("Images:", img_count)
    print("Time taken:", stop - start)
    print("Sec/prediction:", (stop - start) / img_count)
    print("Frame/sec:", img_count / (stop - start))
