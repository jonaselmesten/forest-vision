import datetime
import logging
import time
from collections import deque

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds, log_every_n


class LossEvalHook(HookBase):
    """
    Hook that calculates the evaluation loss for the model.
    """

    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):

        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []

        for idx, inputs in enumerate(self._data_loader):

            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_compute_time += time.perf_counter() - start_compute_time
            iter_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iter_after_start

            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iter_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))

                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)

        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)

        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):

        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter

        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()

        self.trainer.storage.put_scalars(timetest=12)


class BestCheckpoint(HookBase):

    def __init__(self, interval):
        """
        Hook that saves the model if the loss values has improved.
        :param interval: Iteration interval to check for improvement.
        """
        self.interval = interval
        self.total_loss = deque(maxlen=3)
        self.loss_box_reg = deque(maxlen=3)
        self.loss_mask = deque(maxlen=3)
        self.loss_val = deque(maxlen=3)
        self.last_save = 0

    def _is_best(self, tot_loss, box_loss, mask_loss, val_loss):
        """
        Compare the last three times of iteration intervals.
        If one of the stored values has a lower loss value the model will
        not be saved as.
        :return: True if best, false if otherwise.
        """
        if max(self.total_loss) < tot_loss:
            log_every_n(logging.WARNING,
                        "Total loss hasn't improved. Most recent loss:" + str(tot_loss),
                        + "old loss:" + str(max(self.total_loss)))
            return False
        if max(self.loss_box_reg) < box_loss:
            log_every_n(logging.WARNING,
                        "Box loss hasn't improved. Most recent loss:" + str(box_loss),
                        + "old loss:" + str(max(self.loss_box_reg)))
            return False
        if max(self.loss_mask) < mask_loss:
            log_every_n(logging.WARNING,
                        "Mask loss hasn't improved. Most recent loss:" + str(mask_loss),
                        + "old loss:" + str(max(self.loss_mask)))
            return False
        if max(self.loss_val) < val_loss:
            log_every_n(logging.WARNING,
                        "Val loss hasn't improved. Most recent loss:" + str(val_loss),
                        + "old loss:" + str(max(self.loss_val)))
            return False

        return True

    def after_step(self):
        """
        Evaluates the model after x interval steps.
        Adds new loss vales to be compared in future steps.
        """
        next_iter = self.trainer.iter + 1

        if next_iter % self.interval == 0:

            log_every_n(
                logging.INFO,
                "Evaluating before saving best model.",
            )

            metric = self.trainer.storage.latest()

            tot_loss = metric["total_loss"]
            box_loss = metric["loss_box_reg"]
            mask_loss = metric["loss_mask"]
            val_loss = metric["validation_loss"]

            if len(self.total_loss) > 0:
                if self._is_best(tot_loss, box_loss, mask_loss, val_loss):
                    self.trainer.checkpointer.save("model_best")
                    self.last_save = next_iter
                    log_every_n(logging.INFO, "Best model saved.")
                else:
                    log_every_n(logging.WARNING,
                                "Model hasn't improved - Not saving - Last save at iter:" + str(self.last_save))

            self.total_loss.append(metric["total_loss"])
            self.loss_box_reg.append(metric["loss_box_reg"])
            self.loss_mask.append(metric["loss_mask"])
            self.loss_val.append(metric["validation_loss"])
