from airgym.envs.drone_env import AirSimDroneEnv
import airgym

import ray
from ray import tune
from ray.tune.registry import register_env

from ray.rllib.agents import with_common_config


DEFAULT_CONFIG = with_common_config({
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,

    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,

    # The GAE (lambda) parameter.
    "lambda": 1.0,

    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,

    # Size of batches collected from each worker.
    "rollout_fragment_length": 200,

    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 4000,

    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,

    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,

    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 30,

    # Stepsize of SGD.
    "lr": 5e-5,

    # Learning rate schedule.
    "lr_schedule": None,

    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers=True inside your model's config.
    "vf_loss_coeff": 1.0,

    "model": {
        # Share layers for value function. If you set this to True, it's
        # important to tune vf_loss_coeff.
        "vf_share_layers": False,
    },

    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.0,

    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,

    # PPO clip parameter.
    "clip_param": 0.3,

    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,

    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,

    # Target value for KL divergence.
    "kl_target": 0.01,

    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "truncate_episodes",

    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",
})

if __name__ == "__main__":
    ray.init(local_mode=True)

    register_env("drone_env", 
                 lambda config: AirSimDroneEnv("127.0.0.1", 
                 3,
                 (84, 84, 1),
                 useDepth=False))

    DEFAULT_CONFIG["env"] = "drone_env"
    DEFAULT_CONFIG["framework"] = "torch"

    results = tune.run("PPO", config=DEFAULT_CONFIG)