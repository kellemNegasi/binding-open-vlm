"""
Utilities for generating the 3D datasets described in the
"Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem"
paper. The package exposes helpers for sampling object configurations
and orchestrating Blender renders via the CLEVR pipeline.
"""

from . import sampling, devices

__all__ = ["sampling", "devices"]
