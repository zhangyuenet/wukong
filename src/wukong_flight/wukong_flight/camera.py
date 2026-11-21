# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Take photos from a sensor_msgs/Image topic and save them as JPEG.

Uses Pillow rather than OpenCV/cv_bridge: the apt OpenCV is built against
numpy 1.x, while PX4's setup installs numpy 2.x into ~/.local, and the two
crash together. Pillow decodes the raw rgb8/bgr8 bytes directly.
用 Pillow 而不是 OpenCV/cv_bridge：apt 的 OpenCV 依赖 numpy 1.x，而 PX4 的安装
脚本在 ~/.local 装了 numpy 2.x，两者不兼容。
"""

import os
from typing import Callable, Optional

from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

# ROS encoding -> (Pillow raw mode of the bytes, Pillow image mode)
_MODES = {'rgb8': ('RGB', 'RGB'), 'bgr8': ('BGR', 'RGB'),
          'rgba8': ('RGBA', 'RGBA'), 'bgra8': ('BGRA', 'RGBA'), 'mono8': ('L', 'L')}


def image_to_pil(msg: Image) -> PILImage.Image:
    if msg.encoding not in _MODES:
        raise ValueError(f'unsupported image encoding {msg.encoding}')
    raw, mode = _MODES[msg.encoding]
    img = PILImage.frombytes(mode, (msg.width, msg.height), bytes(msg.data),
                             'raw', raw, msg.step)
    return img.convert('RGB')


class Camera:
    """
    Capture the next frame after a request.

    Waiting for a *new* frame, instead of saving the last one received, makes
    sure the photo shows the vehicle where it is now, settled and facing the
    target. 等待请求之后的新一帧，确保照片是飞机就位之后拍的。
    """

    def __init__(self, node: Node, topic: str = '/camera/image_raw') -> None:
        self._node = node
        self._pending: Optional[tuple] = None   # (path, callback)
        self.frames = 0
        node.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)

    def capture(self, path: str, done: Callable[[bool, str], None]) -> None:
        """Save the next frame to path, then call done(ok, path_or_error)."""
        self._pending = (path, done)

    def _on_image(self, msg: Image) -> None:
        self.frames += 1
        if self._pending is None:
            return
        path, done = self._pending
        self._pending = None
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            image_to_pil(msg).save(path, quality=90)
        except (OSError, ValueError) as e:
            done(False, str(e))
            return
        done(True, path)
