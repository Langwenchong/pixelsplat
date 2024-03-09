from typing import Optional

from .encoder import Encoder
from .encoder_epipolar import EncoderEpipolar, EncoderEpipolarCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_epipolar import EncoderVisualizerEpipolar

# 注意这里返还的是一个字典，并且key-value是string-class tuple格式，不是具体的某个实例
ENCODERS = {
    "epipolar": (EncoderEpipolar, EncoderVisualizerEpipolar),
}

EncoderCfg = EncoderEpipolarCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
