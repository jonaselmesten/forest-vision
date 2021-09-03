import time

import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

from model.config import cfg_instance, cfg_semantic
from model.predictor import SemanticInstancePredictor, InstancePredictor, SemanticPredictor
from visualize import CustomVisualizer

video = cv2.VideoCapture("output.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

video_writer = cv2.VideoWriter('forest.mp4',
                               fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                               fps=float(frames_per_second),
                               frameSize=(width, height),
                               isColor=True)


def run_on_video(vid, threshold=0.75, video_length_sec=60, batch_size=1, live_feed=False):
    """
    Writes the inference result to the output video.
    :param vid: Video.
    :param threshold: Accuracy threshold for instance segmentation.
    :param video_length_sec: How long to make the output video.
    :param batch_size: Batch size of inference.
    """
    cfg_instance.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    instance_predictor = InstancePredictor(cfg_instance)
    semantic_predictor = SemanticPredictor(cfg_semantic)

    img_list = []
    frame_count = 0

    print("Starting to write to video...")

    while True:
        has_frame, img = vid.read()
        if not has_frame:
            break

        frame_count += 1
        outputs = instance_predictor(img)
        img_seg = semantic_predictor(img)

        v = CustomVisualizer(img[:, :, ::-1],
                             metadata=MetadataCatalog.get("stem_train"),
                             metadata_semantic=MetadataCatalog.get(cfg_semantic.DATASETS.TRAIN[0]),
                             instance_mode=ColorMode(1))

        out = v.draw_sem_seg(img_seg["sem_seg"].argmax(dim=0).to("cpu"))
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        visualization = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)

        if live_feed:
            cv2.namedWindow("WINDOW_NAME", cv2.WINDOW_NORMAL)
            cv2.imshow("WINDOW_NAME", visualization)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        else:
            video_writer.write(visualization)

run_on_video(video, video_length_sec=10, live_feed=True)

cv2.destroyAllWindows()
video_writer.release()
