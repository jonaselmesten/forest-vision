
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

from annotation import vgg_to_data_dict
from model.loader import load_mask_rcnn_R_50_FPN_3x, load_per_pixel_baseline_plus_R50_bs16_160k, load_pan

setup_logger()

# Setup train/val datasets.
for mode in ["train", "val"]:
    DatasetCatalog.register("stem_" + mode, lambda d=mode: vgg_to_data_dict("stem/" + mode))
    MetadataCatalog.get("stem_" + mode).set(
        thing_colors=[(250, 0, 0), (25, 255, 25), (34, 0, 204), (0, 0, 255), (0, 0, 255), (0, 0, 255)],
        thing_classes=["spruce", "birch", "pine", "spruce-crown", "birch-crown", "pine-crown"])

metadata_train = MetadataCatalog.get("stem_train")

cfg_instance = load_mask_rcnn_R_50_FPN_3x()
#cfg_semantic = load_per_pixel_baseline_plus_R50_bs16_160k()
cfg_semantic = load_pan()

