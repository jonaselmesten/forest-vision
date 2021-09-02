import os
import time

import cv2
import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer, ColorMode

from model import cfg_instance, cfg_semantic
from visualize import CustomVisualizer


class SemanticInstancePredictor:

    def __init__(self, cfg_instance, cfg_semantic):

        self.cfg_instance = cfg_instance.clone()  # cfg can be modified by model
        self.instance_model = build_model(self.cfg_instance)
        self.instance_model.eval()

        if len(cfg_instance.DATASETS.TEST):
            self.metadata_instance = MetadataCatalog.get(cfg_instance.DATASETS.TEST[0])

        checkpointer_instance = DetectionCheckpointer(self.instance_model)
        checkpointer_instance.load(cfg_instance.MODEL.WEIGHTS)

        self.input_format = cfg_instance.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        # ---------------------------------

        self.cfg_semantic = cfg_semantic.clone()  # cfg can be modified by model
        self.semantic_model = build_model(self.cfg_semantic)
        self.semantic_model.eval()

        if len(cfg_semantic.DATASETS.TEST):
            self.metadata_semantic = MetadataCatalog.get(cfg_semantic.DATASETS.TEST[0])

        checkpointer_semantic = DetectionCheckpointer(self.semantic_model)
        checkpointer_semantic.load(cfg_semantic.MODEL.WEIGHTS)

        self.input_format = cfg_semantic.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
        """

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258

            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}

            instance_prediction = self.instance_model([inputs])[0]
            semantic_prediction = self.semantic_model([inputs])[0]

            return instance_prediction, semantic_prediction

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

            instance_prediction = self.instance_model(input_list)[0]
            semantic_prediction = self.semantic_model(input_list)[0]

            return instance_prediction, semantic_prediction


class InstancePredictor:

    def __init__(self, cfg):

        self.cfg = cfg.clone()
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

        self.cfg = cfg.clone()
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




