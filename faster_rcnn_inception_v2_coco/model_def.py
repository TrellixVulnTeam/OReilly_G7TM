"""
Trains a tensorflow object detection model
"""

from typing import Any, Dict

import tensorflow
from packaging import version

from pedl.frameworks.tensorflow.estimator_trial import EstimatorTrial, ServingInputReceiverFn
from pedl.trial import get_trial_seed
from pedl.util import get_experiment_config, get_hyperparameters

from object_detection import model_hparams
from object_detection import model_lib

# Handle TensorFlow compatibility issues.
if version.parse(tensorflow.__version__) >= version.parse("1.14.0"):
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf


class ObjectDetectConfigError(Exception):
    """Raised when object detection config is invalid or missing"""
    pass


def getObjectDetectConfig():
    config = get_experiment_config().get("data", {}).get("object_detection")
    if config is None:
        raise ObjectDetectConfigError("data.object_detection config field is missing")

    if "pipeline_config_path" not in config:
        raise ObjectDetectConfigError("data.object_detection.pipeline_config_path is a required field")

    fields = ["sample_1_of_n_examples", "sample_1_of_n_eval_ examples", "train_steps"]
    for key in fields:
        if key not in config:
            config[key] = 1

    config["hparams"] = model_hparams.create_hparams(None)
    config = {**config, **get_hyperparameters()}

    return config


class ObjectDetectTrial(EstimatorTrial):
    def __init__(self, *args):
        super().__init__(*args)

        config = getObjectDetectConfig()

        self.train_and_eval_dict = model_lib.create_estimator_and_inputs(
          run_config=tf.estimator.RunConfig(tf_random_seed=get_trial_seed()),
          **config)

        train_input_fn = self.train_and_eval_dict['train_input_fn']
        eval_input_fns = self.train_and_eval_dict['eval_input_fns']
        eval_on_train_input_fn = self.train_and_eval_dict['eval_on_train_input_fn']
        predict_input_fn = self.train_and_eval_dict['predict_input_fn']
        train_steps = self.train_and_eval_dict['train_steps']

        self.training_spec, self.validation_spec = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False)

    def build_estimator(self, hparams: Dict[str, Any]) -> tf.estimator.Estimator:
        return self.train_and_eval_dict["estimator"]

    def build_train_spec(self, hparams: Dict[str, Any]) -> tf.estimator.TrainSpec:
        return self.training_spec

    def build_validation_spec(self, hparams: Dict[str, Any]) -> tf.estimator.EvalSpec:
        return self.validation_spec[0]

    def build_serving_input_receiver_fns(self, hparams: Dict[str, Any]) -> Dict[str, ServingInputReceiverFn]:
        return {"serving": self.train_and_eval_dict['predict_input_fn']}
