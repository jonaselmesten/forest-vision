import json
import os

from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from augmentation import image_augmentation
from model import cfg_instance
from hooks import LossEvalHook, BestCheckpoint


class CustomTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=image_augmentation)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()

        hooks.insert(-1, LossEvalHook(

            cfg_instance.TEST.EVAL_PERIOD,
            self.model,

            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))

        hooks.append(BestCheckpoint(cfg_instance.TEST.EVAL_PERIOD))

        return hooks


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


