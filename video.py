import cv2
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import Visualizer, ColorMode

from config import cfg_instance
from main import metadata_train

video = cv2.VideoCapture("forest.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

max_frames = num_frames // 3

video_writer = cv2.VideoWriter('forest_out.mp4',
                               fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                               fps=float(frames_per_second),
                               frameSize=(width, height),
                               isColor=True)

visualizer = VideoVisualizer(metadata=metadata_train)


def run_on_video(vid, max_frame_count, threshold=0.75):
    cfg_instance.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    predictor = DefaultPredictor(cfg_instance)

    read_frames = 0

    while True:

        has_frame, img = vid.read()

        if not has_frame:
            break

        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1],
                       MetadataCatalog.get("stem_train"),
                       instance_mode=ColorMode(1))

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        prediction = out.get_image()[:, :, ::-1]

        yield prediction

        read_frames += 1
        if read_frames > max_frame_count:
            break


count = 0

for frame in run_on_video(video, max_frames):
    print("Frame ", count, " of total:", num_frames // 3)
    count += 1
    video_writer.write(frame)

cv2.destroyAllWindows()
video_writer.release()
