import cv2

from model import cfg_instance, cfg_semantic
from model.predictor import SemanticInstancePredictor

video = cv2.VideoCapture("forest.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

video_writer = cv2.VideoWriter('forest_out60.mp4',
                               fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                               fps=float(frames_per_second),
                               frameSize=(width, height),
                               isColor=True)


def run_on_video(vid, threshold=0.75, video_length_sec=60, batch_size=1):
    """
    Writes the inference result to the output video.
    :param vid: Video.
    :param threshold: Accuracy threshold for instance segmentation.
    :param video_length_sec: How long to make the output video.
    :param batch_size: Batch size of inference.
    """
    cfg_instance.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    predictor = SemanticInstancePredictor(cfg_instance, cfg_semantic)

    img_list = []

    print("Starting to write to video...")

    while True:
        has_frame, img = vid.read()
        if not has_frame:
            return
        else:
            img_list.append(img)
            if len(img_list) == batch_size:
                break

    outputs, semantic = predictor.batch_process(img_list)
    # TODO: Add writer etc.


run_on_video(video, video_length_sec=10)
cv2.destroyAllWindows()
video_writer.release()
