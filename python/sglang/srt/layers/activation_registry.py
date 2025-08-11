# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flexible activation function registry for MoE and other layers."""

import logging
from enum import IntEnum
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.custom_op import CustomOp
from sglang.srt.layers.activation import GeluAndMul, SiluAndMul

logger = logging.getLogger(__name__)


class ActivationType(IntEnum):
    """Enumeration for activation function types."""

    SILU = 0
    GELU = 1
    SWISH = 2
    RELU = 3
    LEAKY_RELU = 4
    GELU_TANH = 5


class ParameterizedActivation(nn.Module):
    """Base class for parameterized activation functions."""

    def __init__(
        self, activation_type: ActivationType, params: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.activation_type = activation_type
        self.params = params or {}

    def get_kernel_params(self) -> Dict[str, float]:
        """Get parameters suitable for kernel calls."""
        return {}


class SiluActivation(ParameterizedActivation):
    """SiLU activation: x * sigmoid(x)"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(ActivationType.SILU, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class GeluActivation(ParameterizedActivation):
    """GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(ActivationType.GELU, params)
        self.approximate = params.get("approximate", "none") if params else "none"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


class SwishActivation(ParameterizedActivation):
    """Swish activation: x * sigmoid(beta * x)"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(ActivationType.SWISH, params)
        self.beta = params.get("beta", 1.0) if params else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)

    def get_kernel_params(self) -> Dict[str, float]:
        return {"beta": self.beta}


class ReluActivation(ParameterizedActivation):
    """ReLU activation: max(0, x)"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(ActivationType.RELU, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class LeakyReluActivation(ParameterizedActivation):
    """Leaky ReLU activation: max(alpha * x, x)"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(ActivationType.LEAKY_RELU, params)
        self.alpha = params.get("alpha", 0.01) if params else 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.alpha)

    def get_kernel_params(self) -> Dict[str, float]:
        return {"alpha": self.alpha}


class GeluTanhActivation(ParameterizedActivation):
    """GELU with tanh approximation"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(ActivationType.GELU_TANH, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


class SwishAndMul(CustomOp):
    """Parameterized Swish GLU: gate = swish(x[:d] * beta), up = x[d:], output = gate * up"""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        return torch.sigmoid(self.beta * gate) * gate * up

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # For now, fall back to native implementation
        # TODO: Implement CUDA kernel for parameterized swish
        return self.forward_native(x)


class ParameterizedGeluAndMul(GeluAndMul):
    """Extended GELU GLU with configurable approximation"""

    def __init__(self, approximate: str = "none"):
        super().__init__(approximate=approximate)


class ActivationRegistry:
    """Registry for activation functions with parameter support."""

    _ACTIVATION_CLASSES = {
        "silu": SiluActivation,
        "gelu": GeluActivation,
        "swish": SwishActivation,
        "relu": ReluActivation,
        "leaky_relu": LeakyReluActivation,
        "gelu_tanh": GeluTanhActivation,
        # Aliases for compatibility
        "swiglu": SiluActivation,  # SwiGLU typically uses SiLU
        "geglu": GeluActivation,  # GEGLU typically uses GELU
    }

    _TYPE_TO_NAME = {
        ActivationType.SILU: "silu",
        ActivationType.GELU: "gelu",
        ActivationType.SWISH: "swish",
        ActivationType.RELU: "relu",
        ActivationType.LEAKY_RELU: "leaky_relu",
        ActivationType.GELU_TANH: "gelu_tanh",
    }

    @classmethod
    def get_activation(
        cls, name: str, params: Optional[Dict[str, Any]] = None
    ) -> ParameterizedActivation:
        """Get activation function by name with optional parameters."""
        name = name.lower().strip()

        if name not in cls._ACTIVATION_CLASSES:
            raise ValueError(
                f"Unsupported activation function: {name}. "
                f"Supported: {list(cls._ACTIVATION_CLASSES.keys())}"
            )

        activation_class = cls._ACTIVATION_CLASSES[name]
        return activation_class(params)

    @classmethod
    def get_activation_type(cls, name: str) -> ActivationType:
        """Get activation type enum by name."""
        activation = cls.get_activation(name)
        return activation.activation_type

    @classmethod
    def get_name_by_type(cls, activation_type: ActivationType) -> str:
        """Get activation name by type enum."""
        return cls._TYPE_TO_NAME[activation_type]

    @classmethod
    def get_fused_activation(
        cls, name: str, params: Optional[Dict[str, Any]] = None
    ) -> Union[SiluAndMul, GeluAndMul, SwishAndMul, ParameterizedGeluAndMul]:
        """Get fused activation function for GLU-style operations."""
        name = name.lower().strip()
        params = params or {}

        if name in ["silu", "swiglu"]:
            return SiluAndMul()
        elif name == "swish":
            beta = params.get("beta", 1.0)
            if beta == 1.0:
                # Swish with beta=1.0 is equivalent to SiLU
                return SiluAndMul()
            else:
                return SwishAndMul(beta=beta)
        elif name in ["gelu", "geglu"]:
            approximate = params.get("approximate", "none")
            return ParameterizedGeluAndMul(approximate=approximate)
        elif name == "gelu_tanh":
            return ParameterizedGeluAndMul(approximate="tanh")
        else:
            raise ValueError(
                f"Unsupported fused activation: {name}. "
                "Supported: silu, swiglu, swish, gelu, geglu, gelu_tanh"
            )

    @classmethod
    def is_supported(cls, name: str) -> bool:
        """Check if activation function is supported."""
        return name.lower().strip() in cls._ACTIVATION_CLASSES

    @classmethod
    def list_supported(cls) -> list:
        """List all supported activation functions."""
        return list(cls._ACTIVATION_CLASSES.keys())


def parse_activation_config(config: Dict[str, Any]) -> tuple:
    """
    Parse activation configuration from model config.

    Returns:
        tuple: (activation_name, activation_params)
    """
    # Support multiple config key names for compatibility
    activation_keys = [
        "hidden_act",  # Standard HuggingFace key
        "activation_function",  # Alternative key
        "hidden_activation",  # GLM style
        "activation",  # Simple key
    ]

    param_keys = [
        "activation_params",  # Our standard key
        "activation_kwargs",  # Alternative key
        "act_params",  # Short key
    ]

    # Find activation function name
    activation_name = None
    for key in activation_keys:
        if key in config:
            activation_name = config[key]
            break

    if activation_name is None:
        activation_name = "silu"  # Default

    # Find activation parameters
    activation_params = {}
    for key in param_keys:
        if key in config:
            activation_params = config[key]
            break

    return activation_name, activation_params


def get_default_activation_for_model(model_type: str) -> tuple:
    """
    Get default activation function for specific model types.

    Returns:
        tuple: (activation_name, activation_params)
    """
    model_type = model_type.lower()

    # Default configurations for different model families
    defaults = {
        # Llama family
        "llama": ("silu", {}),
        "llamaforcausallm": ("silu", {}),
        # GPT family
        "gpt": ("gelu", {}),
        "gpt2": ("gelu", {}),
        "gptneox": ("gelu", {}),
        # Mistral family
        "mistral": ("silu", {}),
        "mistralmodel": ("silu", {}),
        "mistralforcausallm": ("silu", {}),
        "mixtral": ("silu", {}),
        "mixtralforcausallm": ("silu", {}),
        # DeepSeek family
        "deepseek": ("silu", {}),
        "deepseekmoe": ("silu", {}),
        "deepseekv2": ("silu", {}),
        "deepseekv3": ("silu", {}),
        # Qwen family
        "qwen": ("silu", {}),
        "qwen2": ("silu", {}),
        "qwen2.5": ("silu", {}),
        "qwen3": ("silu", {}),
        "qwenmoe": ("silu", {}),
        # GLM family (typically GELU)
        "glm": ("gelu", {}),
        "chatglm": ("gelu", {}),
        "glm4": ("gelu", {}),
        # Other families
        "gemma": ("gelu", {"approximate": "tanh"}),
        "phi": ("gelu", {"approximate": "tanh"}),
        "baichuan": ("silu", {}),
        "internlm": ("silu", {}),
        "dbrx": ("silu", {}),
        "granite": ("silu", {}),
    }

    for pattern, config in defaults.items():
        if pattern in model_type:
            return config

    # Default fallback
    return ("silu", {})
