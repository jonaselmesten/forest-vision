import os

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger

from annotation import vgg_to_data_dict
from config import get_config_file, get_model_weights

setup_logger()


# Semantic ---------------------------------------------------------------
cfg_semantic = get_cfg()
cfg_semantic.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg_semantic.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
cfg_semantic.MODEL.DEVICE = "cuda"

# Instance ---------------------------------------------------------------

# Setup train/val datasets.
for mode in ["train", "val"]:
    DatasetCatalog.register("stem_" + mode, lambda d=mode: vgg_to_data_dict("stem/" + mode))
    MetadataCatalog.get("stem_" + mode).set(
        thing_colors=[(250, 0, 0), (25, 255, 25), (34, 0, 204), (0, 0, 255), (0, 0, 255), (0, 0, 255)],
        thing_classes=["spruce", "birch", "pine", "spruce-crown", "birch-crown", "pine-crown"])

metadata_train = MetadataCatalog.get("stem_train")

cfg_instance = get_cfg()
cfg_instance.merge_from_file(get_config_file("instance/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_instance.DATASETS.TRAIN = ("stem_train",)
cfg_instance.DATASETS.TEST = ("stem_val",)
cfg_instance.TEST.EVAL_PERIOD = 10
cfg_instance.MODEL.DEVICE = "cuda"

cfg_instance.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg_instance.MODEL.WEIGHTS = get_model_weights("model_best_4500.pth")
cfg_instance.MODEL.ROI_HEADS.NUM_CLASSES = 6

cfg_instance.SOLVER.IMS_PER_BATCH = 1
cfg_instance.SOLVER.BASE_LR = 0.00025
cfg_instance.SOLVER.MAX_ITER = 300
cfg_instance.SOLVER.STEPS = []
cfg_instance.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
