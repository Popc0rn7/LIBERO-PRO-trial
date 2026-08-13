"""Single Hydra CLI entry point for policy evaluation."""

import hydra
from omegaconf import DictConfig, OmegaConf

from .clients import create_client
from .protocol import ActionSpec, PROTOCOL_VERSION
from .runner import EvaluationRunner


def validate_config(cfg):
    if cfg.policy.protocol.version != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version {!r}".format(cfg.policy.protocol.version))
    spec = ActionSpec(**OmegaConf.to_container(cfg.policy.action, resolve=True))
    if spec.dim != 7 or spec.type != "delta_ee":
        raise ValueError("LIBERO requires delta_ee actions with dim=7")
    for field in ("execute_horizon", "max_steps", "episode_timeout_seconds"):
        if float(cfg.rollout[field]) <= 0:
            raise ValueError("rollout.{} must be positive".format(field))
    if int(cfg.rollout.warmup_steps) < 0:
        raise ValueError("rollout.warmup_steps cannot be negative")


def run(cfg):
    validate_config(cfg)
    client = create_client(cfg.policy)
    try:
        video_writer = None
        if cfg.recording.save_video:
            from libero.libero.utils.video_utils import VideoWriter
            video_writer = VideoWriter(
                str(cfg.recording.video_directory), True,
                fps=int(cfg.recording.fps), single_video=False
            )
        try:
            return EvaluationRunner(cfg, client, video_writer=video_writer).run()
        finally:
            if video_writer is not None:
                video_writer.save()
    finally:
        client.close()


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig):
    summary, _ = run(cfg)
    print(OmegaConf.to_yaml(summary))


if __name__ == "__main__":
    main()
