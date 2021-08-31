import os

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger

from annotation import vgg_to_data_dict

setup_logger()

# Setup train/val datasets.
for mode in ["train", "val"]:
    DatasetCatalog.register("stem_" + mode, lambda d=mode: vgg_to_data_dict("stem/" + mode))
    MetadataCatalog.get("stem_" + mode).set(
        thing_colors=[(250, 0, 0), (25, 255, 25), (34, 0, 204), (0, 0, 255), (0, 0, 255), (0, 0, 255)],
        thing_classes=["spruce", "birch", "pine", "spruce-crown", "birch-crown", "pine-crown"])

metadata_train = MetadataCatalog.get("stem_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("stem_train",)
cfg.DATASETS.TEST = ("stem_val",)
cfg.TEST.EVAL_PERIOD = 150

cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 2500
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
