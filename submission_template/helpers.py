import ctypes
import math
import os
from enum import Enum

import torch
import torchvision.transforms.functional as functional
from torchvision.transforms import InterpolationMode


MEBIBYTE = 1024 ** 2


class AugmentationType(Enum):
    TRANSLATION = "translation"
    PIXEL_NOISE = "pixel_noise"
    OCCLUSION = "occlusion"
    ROTATION = "rotation"
    FLIP = "flip"
    RGB = "rgb"


class FlipDirection(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class ImageAugmentations:
    @staticmethod
    def translate(
        image,
        horizontal_percent,
        vertical_percent,
        fill_value=0.0,
        interpolation=InterpolationMode.BILINEAR,
    ):
        if abs(horizontal_percent) > 100 or abs(vertical_percent) > 100:
            raise ValueError("translation percentages must be between -100 and 100")

        height, width = image.shape[-2:]
        horizontal_pixels = round(width * horizontal_percent / 100.0)
        vertical_pixels = round(height * vertical_percent / 100.0)

        return functional.affine(
            image,
            angle=0.0,
            translate=[horizontal_pixels, vertical_pixels],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=interpolation,
            fill=fill_value,
        )

    @staticmethod
    def add_pixel_noise(image, noise_std_percent, generator=None):
        if noise_std_percent < 0:
            raise ValueError("noise_std_percent must be non-negative")

        channel_std = image.std(dim=(-2, -1), unbiased=False, keepdim=True)
        noise_scale = channel_std * noise_std_percent / 100.0
        noise = torch.randn(
            image.shape,
            dtype=image.dtype,
            device=image.device,
            generator=generator,
        )
        return image + noise * noise_scale

    @staticmethod
    def occlude(
        image,
        area_percent,
        aspect_ratio=1.0,
        fill_value=0.0,
        generator=None,
    ):
        if not 0.0 <= area_percent <= 100.0:
            raise ValueError("area_percent must be between 0 and 100")
        if aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be positive")
        if area_percent == 0:
            return image

        height, width = image.shape[-2:]
        target_area = height * width * area_percent / 100.0
        occlusion_height = min(height, max(1, round(math.sqrt(target_area / aspect_ratio))))
        occlusion_width = min(width, max(1, round(math.sqrt(target_area * aspect_ratio))))
        top = torch.randint(
            0,
            height - occlusion_height + 1,
            (1,),
            generator=generator,
        ).item()
        left = torch.randint(
            0,
            width - occlusion_width + 1,
            (1,),
            generator=generator,
        ).item()

        result = image.clone()
        result[:, top:top + occlusion_height, left:left + occlusion_width] = fill_value
        return result

    @staticmethod
    def rotate(
        image,
        degrees,
        fill_value=0.0,
        interpolation=InterpolationMode.BILINEAR,
    ):
        if not -180.0 <= degrees <= 180.0:
            raise ValueError("degrees must be between -180 and 180")

        return functional.rotate(
            image,
            angle=degrees,
            interpolation=interpolation,
            fill=fill_value,
        )

    @staticmethod
    def flip(image, direction):
        dimensions = {
            FlipDirection.HORIZONTAL: -1,
            FlipDirection.VERTICAL: -2,
        }
        dimension = dimensions[direction]
        return torch.flip(image, dims=(dimension,))

    @staticmethod
    def adjust_rgb(
        image,
        value_range,
        brightness_percent=0.0,
        contrast_percent=0.0,
        saturation_percent=0.0,
        hue_degrees=0.0,
    ):
        if image.shape[-3] != 3:
            raise ValueError("RGB adjustments require exactly three channels")
        if not all(-100.0 <= value <= 100.0 for value in (
            brightness_percent,
            contrast_percent,
            saturation_percent,
        )):
            raise ValueError("RGB percentages must be between -100 and 100")
        if not -180.0 <= hue_degrees <= 180.0:
            raise ValueError("hue_degrees must be between -180 and 180")

        value_min, value_max = value_range
        if value_min >= value_max:
            raise ValueError("value_range must be an ascending pair")

        normalized = ((image - value_min) / (value_max - value_min)).clamp(0.0, 1.0)
        normalized = functional.adjust_brightness(normalized, 1.0 + brightness_percent / 100.0)
        normalized = functional.adjust_contrast(normalized, 1.0 + contrast_percent / 100.0)
        normalized = functional.adjust_saturation(normalized, 1.0 + saturation_percent / 100.0)
        normalized = functional.adjust_hue(normalized, hue_degrees / 360.0)
        return normalized.clamp(0.0, 1.0) * (value_max - value_min) + value_min


def get_available_cpu_memory():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available_physical)

    if hasattr(os, "sysconf"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)

    return None


def get_hardware_info():
    hardware = {
        "device_type": "cpu",
        "cpu_count": os.cpu_count(),
        "cpu_available_memory_bytes": get_available_cpu_memory(),
        "gpu_name": None,
        "gpu_total_memory_bytes": None,
    }

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        hardware.update({
            "device_type": "cuda",
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
        })

    return hardware


def choose_batch_size(input_shape, hardware, max_batch_size=64):
    _, channels, height, width = input_shape
    bytes_per_sample = channels * height * width * 4  # float32

    memory_limits = [
        memory
        for memory in (
            hardware["cpu_available_memory_bytes"],
            hardware["gpu_total_memory_bytes"],
        )
        if memory is not None
    ]
    relevant_memory = min(memory_limits) if memory_limits else None

    if relevant_memory is None:
        input_batch_budget = 32 * MEBIBYTE
    else:
        input_batch_budget = min(64 * MEBIBYTE, int(relevant_memory * 0.005))

    candidate = min(max_batch_size, max(1, input_batch_budget // bytes_per_sample))
    return 1 << (candidate.bit_length() - 1)
