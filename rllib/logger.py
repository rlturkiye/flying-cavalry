from typing import List

from ray.tune.logger import Logger
import csv
import json
import logging
import os
import yaml
import numbers
import numpy as np


from ray.rllib.utils.framework import try_import_torch
import ray.cloudpickle as cloudpickle
from ray.util.debug import log_once
from ray.tune.result import (NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S,
                             TIMESTEPS_TOTAL, EXPR_PARAM_FILE,
                             EXPR_PARAM_PICKLE_FILE, EXPR_PROGRESS_FILE,
                             EXPR_RESULT_FILE)
from ray.tune.syncer import get_node_syncer
from ray.tune.utils import flatten_dict
from ray.rllib.utils.typing import TensorType, TrainerConfigDict

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)

VALID_SUMMARY_TYPES = [int, float, np.float32, np.float64, np.int32, np.int64]

class CustomTBXLogger(Logger):
  """TensorBoardX Logger.

  Note that hparams will be written only after a trial has terminated.
  This logger automatically flattens nested dicts to show on TensorBoard:

      {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}
  """
  
  # NoneType is not supported on the last TBX release yet.
  VALID_HPARAMS = (str, bool, np.bool8, int, np.integer, float, list)
  
  def _init(self):
    try:
      from tensorboardX import SummaryWriter
    except ImportError:
      if log_once("tbx-install"):
        logger.info(
          "pip install 'ray[tune]' to see TensorBoard files.")
      raise
    self._file_writer = SummaryWriter(self.logdir, flush_secs=30)
    self.last_result = None
  
  def on_result(self, result, sub_path: List[str] = []):
    step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
    
    tmp = result.copy()
    for k in [
      "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
    ]:
      if k in tmp:
        del tmp[k]  # not useful to log these
    
    flat_result = flatten_dict(tmp, delimiter="/")
    path = ["ray"] + sub_path
    valid_result = {}
    
    for attr, value in flat_result.items():
      # print(attr, value)
      full_attr = "/".join(path + [attr])
      if type(value) in VALID_SUMMARY_TYPES and not np.isnan(value):
        valid_result[full_attr] = value
        self._file_writer.add_scalar(
          full_attr, value, global_step=step)
      elif (type(value) == list
            and len(value) > 0) or (type(value) == np.ndarray
                                    and value.size > 0):
        valid_result[full_attr] = value
        
        # Must be an image
        if type(value) == np.ndarray and len(value.shape) == 3:
          self._file_writer.add_image(
            full_attr, value, global_step=step, dataformats="HWC")
          continue
        # Must be video
        if type(value) == np.ndarray and value.ndim == 5:
          self._file_writer.add_video(
            full_attr, value, global_step=step, fps=20)
          continue
        
        try:
          self._file_writer.add_histogram(
            full_attr, value, global_step=step)
        # In case TensorboardX still doesn't think it's a valid value
        # (e.g. `[[]]`), warn and move on.
        except (ValueError, TypeError):
          if log_once("invalid_tbx_value"):
            logger.warning(
              "You are trying to log an invalid value ({}={}) "
              "via {}!".format(full_attr, value,
                               type(self).__name__))
    self.last_result = valid_result
    self._file_writer.flush()
  
  def flush(self):
    if self._file_writer is not None:
      self._file_writer.flush()
  
  def close(self):
    if self._file_writer is not None:
      if self.trial and self.trial.evaluated_params and self.last_result:
        flat_result = flatten_dict(self.last_result, delimiter="/")
        scrubbed_result = {
          k: value
          for k, value in flat_result.items()
          if type(value) in VALID_SUMMARY_TYPES
        }
        self._try_log_hparams(scrubbed_result)
      self._file_writer.close()
  
  def _try_log_hparams(self, result):
    # TBX currently errors if the hparams value is None.
    flat_params = flatten_dict(self.trial.evaluated_params)
    scrubbed_params = {
      k: v
      for k, v in flat_params.items()
      if isinstance(v, self.VALID_HPARAMS)
    }
    
    removed = {
      k: v
      for k, v in flat_params.items()
      if not isinstance(v, self.VALID_HPARAMS)
    }
    if removed:
      logger.info(
        "Removed the following hyperparameter values when "
        "logging to tensorboard: %s", str(removed))
    
    from tensorboardX.summary import hparams
    try:
      experiment_tag, session_start_tag, session_end_tag = hparams(
        hparam_dict=scrubbed_params, metric_dict=result)
      self._file_writer.file_writer.add_summary(experiment_tag)
      self._file_writer.file_writer.add_summary(session_start_tag)
      self._file_writer.file_writer.add_summary(session_end_tag)
    except Exception:
      logger.exception("TensorboardX failed to log hparams. "
                       "This may be due to an unsupported type "
                       "in the hyperparameter values.")