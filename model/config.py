import random

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

from annotation import vgg_to_data_dict
from model.loader import load_mask_rcnn_R_50_FPN_3x, load_panoptic_fpn_R_101_3x, \
    load_per_pixel_baseline_plus_R50_bs16_160k

setup_logger()


def random_color():
    r = random.randrange(0, 255)
    g = random.randrange(0, 255)
    b = random.randrange(0, 255)

    return tuple([r, g, b])


stuff_colors = {"tree": (55, 100, 10),
                "plant": (107, 246, 72),
                "water": (30, 113, 236),
                "earth, ground": (172, 133, 43),
                "sky": (192, 217, 250),
                "grass": (78, 255, 1),
                "river": (6, 53, 183),
                "road": (92, 92, 100),
                "rock, stone": (142, 144, 149)}

# Setup train/val datasets.
for mode in ["train", "val"]:
    DatasetCatalog.register("stem_" + mode, lambda d=mode: vgg_to_data_dict("stem/" + mode))
    MetadataCatalog.get("stem_" + mode).set(
        thing_colors=[(250, 0, 0), (25, 255, 25), (34, 0, 204), (0, 0, 255), (0, 0, 255), (0, 0, 255)],
        thing_classes=["spruce", "birch", "pine", "spruce-crown", "birch-crown", "pine-crown"])

    MetadataCatalog.get("ade20k_sem_seg_" + mode).set(
        stuff_colors=stuff_colors)

metadata_train = MetadataCatalog.get("stem_train")

cfg_instance = load_mask_rcnn_R_50_FPN_3x()
cfg_semantic = load_per_pixel_baseline_plus_R50_bs16_160k()
# cfg_semantic = load_panoptic_fpn_R_101_3x()
