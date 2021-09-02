import os

from matplotlib import pyplot as plt

from model.config import cfg_instance
from model.trainer import CustomTrainer, load_json_arr
from demo import run_instance_prediction_on_dir, run_semantic_prediction_on_dir, run_semantic_instance_prediction, \
    run_semantic_instance_prediction_on_dir


def show_train_graph():
    experiment_metrics = load_json_arr(cfg_instance.OUTPUT_DIR + '/metrics.json')

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title('Training val/loss')
    ax1.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    ax1.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])

    ax1.legend(['total_loss', 'validation_loss', 'loss_mask'], loc='upper left')

    ax1.set_title('Mask/BBox loss')
    ax2.plot(
        [x['iteration'] for x in experiment_metrics if 'loss_mask' in x],
        [x['loss_mask'] for x in experiment_metrics if 'loss_mask' in x])
    ax2.plot(
        [x['iteration'] for x in experiment_metrics if 'loss_box_reg' in x],
        [x['loss_box_reg'] for x in experiment_metrics if 'loss_box_reg' in x])

    ax2.legend(['loss_mask', 'loss_box'], loc='upper left')

    plt.show()


def train(show_graphs=True):
    os.makedirs(cfg_instance.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg_instance)
    trainer.resume_or_load(resume=False)
    trainer.train()
    show_train_graph()



#train()
#run_instance_batch_prediction("stem/train", num_of_img=4, num_of_cycles=3)

#run_instance_prediction_on_dir("stem/val")
#run_semantic_prediction_on_dir("stem/val")
run_semantic_instance_prediction_on_dir("stem/val")

# vgg_val_split("imgs", "stem/train", "stem/val", "imgs/data.json", 0.2)


