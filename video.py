import time

import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer

from config import cfg
from main import metadata_train

video = cv2.VideoCapture("for.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

video_writer = cv2.VideoWriter('out.mp4',
                               fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                               fps=float(frames_per_second),
                               frameSize=(width, height),
                               isColor=True)

num_frames = 300
write_to_file = False
visualizer = VideoVisualizer(metadata=metadata_train)


def run_on_video(vid, max_frames, threshold=0.6):

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    predictor = DefaultPredictor(cfg)

    read_frames = 0

    while True:

        start_time = time.time()

        has_frame, img = vid.read()

        if not has_frame:
            break

        outputs = predictor(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        prediction = visualizer.draw_instance_predictions(img, outputs["instances"].to("cpu"))

        # Convert Matplotlib RGB format to OpenCV BGR format
        prediction = cv2.cvtColor(prediction.get_image(), cv2.COLOR_RGB2BGR)

        print("--- %s seconds ---" % (time.time() - start_time))
        yield prediction

        read_frames += 1
        if read_frames > max_frames:
            break


for frame in run_on_video(video, num_frames):

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) == 27:
        break
    if write_to_file:
        video_writer.write(frame)

video.release()
video_writer.release()
cv2.destroyAllWindows()
