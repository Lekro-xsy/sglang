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
"""Enhanced Activation Function Registry for MoE layers."""

import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import is_cuda, is_hip, is_npu

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if _is_cuda or _is_hip:
    from sgl_kernel import silu_and_mul, gelu_and_mul, gelu_tanh_and_mul


class SwishActivation(CustomOp):
    """Swish activation function with configurable beta parameter.
    
    Swish(x) = x * sigmoid(beta * x)
    When beta=1.0, this is equivalent to SiLU.
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)
    
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if self.beta == 1.0:
            # Use optimized SiLU kernel when beta=1.0
            return F.silu(x)
        else:
            # Fallback to native implementation for non-standard beta
            return self.forward_native(x)
    
    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda(x)
    
    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)


class SwishAndMul(CustomOp):
    """Swish activation with gated multiplication for MoE layers.
    
    Applies Swish(x[:d]) * x[d:] where d = x.shape[-1] // 2
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d] * torch.sigmoid(self.beta * x[..., :d])
        return gate * x[..., d:]
    
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if self.beta == 1.0:
            # Use optimized SiLU kernel when beta=1.0
            d = x.shape[-1] // 2
            output_shape = x.shape[:-1] + (d,)
            out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            silu_and_mul(x, out)
            return out
        else:
            # Fallback to native implementation for non-standard beta
            return self.forward_native(x)
    
    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda(x)
    
    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)


class LeakyReLUAndMul(CustomOp):
    """LeakyReLU activation with gated multiplication for MoE layers."""
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = F.leaky_relu(x[..., :d], negative_slope=self.negative_slope)
        return gate * x[..., d:]
    
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
    
    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
    
    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)


class ReLUAndMul(CustomOp):
    """ReLU activation with gated multiplication for MoE layers."""
    
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = F.relu(x[..., :d])
        return gate * x[..., d:]
    
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
    
    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
    
    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)


class ActivationRegistry:
    """Enhanced activation function registry with parameter support."""
    
    def __init__(self):
        self._activation_functions: Dict[str, Callable] = {}
        self._gated_activation_functions: Dict[str, Callable] = {}
        self._register_default_activations()
    
    def _register_default_activations(self):
        """Register default activation functions and their gated versions."""
        
        # Standard activation functions
        self._activation_functions.update({
            "silu": lambda **kwargs: SwishActivation(beta=1.0),
            "swish": lambda beta=1.0, **kwargs: SwishActivation(beta=beta),
            "gelu": lambda approximate="none", **kwargs: nn.GELU(approximate=approximate),
            "gelu_tanh": lambda **kwargs: nn.GELU(approximate="tanh"),
            "relu": lambda **kwargs: nn.ReLU(),
            "leaky_relu": lambda negative_slope=0.01, **kwargs: nn.LeakyReLU(negative_slope=negative_slope),
        })
        
        # Gated activation functions for MoE layers (activation + multiplication)
        # Import existing optimized implementations
        from sglang.srt.layers.activation import SiluAndMul, GeluAndMul
        
        self._gated_activation_functions.update({
            "silu": lambda **kwargs: SiluAndMul(),
            "swish": lambda beta=1.0, **kwargs: SwishAndMul(beta=beta),
            "gelu": lambda approximate="none", **kwargs: GeluAndMul(approximate=approximate),
            "gelu_tanh": lambda **kwargs: GeluAndMul(approximate="tanh"),
            "relu": lambda **kwargs: ReLUAndMul(),
            "leaky_relu": lambda negative_slope=0.01, **kwargs: LeakyReLUAndMul(negative_slope=negative_slope),
        })
    
    def register_activation(
        self, 
        name: str, 
        activation_fn: Callable,
        gated_activation_fn: Optional[Callable] = None
    ):
        """Register a new activation function.
        
        Args:
            name: Name of the activation function
            activation_fn: Factory function that creates the activation instance
            gated_activation_fn: Optional factory for gated version (for MoE)
        """
        self._activation_functions[name.lower()] = activation_fn
        if gated_activation_fn is not None:
            self._gated_activation_functions[name.lower()] = gated_activation_fn
        logger.info(f"Registered activation function: {name}")
    
    def get_activation(
        self, 
        name: str, 
        params: Optional[Dict[str, Any]] = None,
        gated: bool = False
    ) -> nn.Module:
        """Get an activation function by name with optional parameters.
        
        Args:
            name: Name of the activation function
            params: Optional parameters for the activation function
            gated: Whether to return gated version (for MoE layers)
            
        Returns:
            Configured activation function instance
            
        Raises:
            ValueError: If activation function is not registered
        """
        name = name.lower()
        params = params or {}
        
        # Choose the appropriate registry
        if gated:
            registry = self._gated_activation_functions
            if name not in registry:
                raise ValueError(f"Gated activation function '{name}' is not registered. "
                               f"Available: {list(registry.keys())}")
        else:
            registry = self._activation_functions
            if name not in registry:
                raise ValueError(f"Activation function '{name}' is not registered. "
                               f"Available: {list(registry.keys())}")
        
        try:
            # Create activation function with parameters
            activation_fn = registry[name](**params)
            logger.debug(f"Created {'gated ' if gated else ''}activation function: {name} with params: {params}")
            return activation_fn
        except Exception as e:
            raise ValueError(f"Failed to create activation function '{name}' with params {params}: {e}")
    
    def list_activations(self, gated: bool = False) -> list:
        """List available activation functions.
        
        Args:
            gated: Whether to list gated versions
            
        Returns:
            List of available activation function names
        """
        registry = self._gated_activation_functions if gated else self._activation_functions
        return list(registry.keys())
    
    def is_registered(self, name: str, gated: bool = False) -> bool:
        """Check if an activation function is registered.
        
        Args:
            name: Name of the activation function
            gated: Whether to check gated version
            
        Returns:
            True if the activation function is registered
        """
        name = name.lower()
        registry = self._gated_activation_functions if gated else self._activation_functions
        return name in registry


# Global activation registry instance
_activation_registry = ActivationRegistry()


def get_activation_function(
    name: str,
    params: Optional[Dict[str, Any]] = None,
    gated: bool = False
) -> nn.Module:
    """Get an activation function from the global registry.
    
    Args:
        name: Name of the activation function
        params: Optional parameters for the activation function
        gated: Whether to return gated version (for MoE layers)
        
    Returns:
        Configured activation function instance
    """
    return _activation_registry.get_activation(name, params, gated)


def register_activation_function(
    name: str,
    activation_fn: Callable,
    gated_activation_fn: Optional[Callable] = None
):
    """Register a new activation function in the global registry.
    
    Args:
        name: Name of the activation function
        activation_fn: Factory function that creates the activation instance
        gated_activation_fn: Optional factory for gated version (for MoE)
    """
    _activation_registry.register_activation(name, activation_fn, gated_activation_fn)


def list_activation_functions(gated: bool = False) -> list:
    """List available activation functions in the global registry.
    
    Args:
        gated: Whether to list gated versions
        
    Returns:
        List of available activation function names
    """
    return _activation_registry.list_activations(gated)


def is_activation_registered(name: str, gated: bool = False) -> bool:
    """Check if an activation function is registered in the global registry.
    
    Args:
        name: Name of the activation function
        gated: Whether to check gated version
        
    Returns:
        True if the activation function is registered
    """
    return _activation_registry.is_registered(name, gated)


# Convenience function for MoE layers
def get_moe_activation_function(
    name: str,
    params: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """Get a gated activation function for MoE layers.
    
    This is a convenience wrapper that automatically sets gated=True.
    
    Args:
        name: Name of the activation function
        params: Optional parameters for the activation function
        
    Returns:
        Configured gated activation function instance
    """
    return get_activation_function(name, params, gated=True)