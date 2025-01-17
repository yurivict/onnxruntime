# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
ONNX Runtime is a performance-focused scoring engine for Open Neural Network Exchange (ONNX) models.
For more information on ONNX Runtime, please see `aka.ms/onnxruntime <https://aka.ms/onnxruntime/>`_ or the `Github project <https://github.com/microsoft/onnxruntime/>`_.
"""
__version__ = "0.5.0"
__author__ = "Microsoft"

from onnxruntime.capi._pybind_state import get_all_providers, get_available_providers, get_device, RunOptions, SessionOptions, set_default_logger_severity, NodeArg, ModelMetadata, GraphOptimizationLevel, ExecutionMode
from onnxruntime.capi.session import InferenceSession
from onnxruntime.capi import onnxruntime_validation
onnxruntime_validation.check_distro_info()
