import os
import platform
import shutil
from typing import Literal

DeviceChoice = Literal["auto", "cpu", "cuda", "optix", "hip", "metal", "oneapi"]

DEVICE_ALIASES = {
    "gpu": "cuda",
    "xpu": "oneapi",
}

def _has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def detect_device(preference: DeviceChoice = "auto") -> str:
    """
    Resolve the preferred compute backend for Blender.
    Preference may be `auto`, `cpu`, `cuda`, `optix`, `hip`, `metal`, or `oneapi`.
    """
    choice = DEVICE_ALIASES.get(preference, preference)
    if choice != "auto":
        return choice

    if _has_cmd("nvidia-smi"):
        return "cuda"
    if _has_cmd("rocminfo"):
        return "hip"
    if _has_cmd("sycl-ls") or os.environ.get("ONEAPI_DEVICE_SELECTOR"):
        return "oneapi"
    if platform.system() == "Darwin":
        return "metal"
    return "cpu"


def cycles_device_flag(choice: str) -> str:
    """
    Map a user-friendly device label to the string expected by Blender's Cycles preferences.
    """
    mapping = {
        "cpu": "CPU",
        "cuda": "CUDA",
        "optix": "OPTIX",
        "hip": "HIP",
        "metal": "METAL",
        "oneapi": "ONEAPI",
    }
    return mapping.get(choice.lower(), "CPU")
