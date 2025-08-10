# SGLang MoE 激活函数灵活化实现说明

## 🎯 实现目标

为 SGLang 的 MoE（专家混合模型）系统添加灵活的激活函数支持，解决硬编码激活函数的限制。

## 🔧 核心修改

### 1. 创建激活函数注册系统
**新增文件**: `python/sglang/srt/layers/activation_registry.py`

```python
# 核心功能
class ActivationRegistry:
    @classmethod
    def get_activation(cls, name: str, params: Optional[Dict] = None) -> ParameterizedActivation
    
    @classmethod
    def get_fused_activation(cls, name: str, params: Optional[Dict] = None) -> Union[SiluAndMul, GeluAndMul, ...]

# 支持的激活函数
- silu: x * sigmoid(x)
- gelu: GELU激活函数，支持 approximate="tanh"
- swish: x * sigmoid(beta * x)，可配置 beta 参数
- relu: max(0, x)
- leaky_relu: max(alpha * x, x)，可配置 alpha 参数
- gelu_tanh: GELU的tanh近似版本
```

### 2. 修改模型配置解析
**修改文件**: `python/sglang/srt/configs/model_config.py`

```python
# 在 ModelConfig 类中添加
def _parse_activation_config(self) -> None:
    """解析激活函数配置，支持多种键名格式"""
    activation_name, activation_params = parse_activation_config(combined_config)
    self.hidden_act = activation_name
    self.activation_params = activation_params

# 支持的配置键名（优先级递减）
activation_keys = [
    "hidden_act",           # HuggingFace 标准
    "activation_function",  # 替代键
    "hidden_activation",    # GLM 风格
    "activation",           # 简单键
]
```

### 3. 更新 Triton MoE 层
**修改文件**: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`

```python
# FusedMoE 类中添加
def _setup_activation_config(self):
    """设置激活函数配置"""
    from sglang.srt.layers.activation_registry import ActivationRegistry
    self.activation_fn = ActivationRegistry.get_activation(self.activation, {})
    self.activation_type = self.activation_fn.activation_type
    self.activation_params = self.activation_fn.get_kernel_params()
```

**修改文件**: `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`

```python
# 内核函数签名更新
def triton_kernel_moe_forward(
    # ... 其他参数
    activation_type: int = 0,
    activation_params: Optional[Dict] = None,
):
    # constexpr 专门化分发
    if activation_type == ActivationType.SILU:
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
    elif activation_type == ActivationType.SWISH:
        beta = activation_params["beta"]
        # 自定义 swish 实现
```

### 4. 更新 Cutlass MoE 层
**修改文件**: `python/sglang/srt/layers/moe/cutlass_moe.py`

```python
# 函数签名更新
def cutlass_fused_experts_fp8(
    # ... 其他参数
    activation_type: int = 0,
    activation_params: Optional[Dict] = None,
):
    # launch-time 分发
    if activation_type == 0:  # SiLU
        silu_and_mul(c1, intermediate)
    elif activation_type == 2:  # Swish
        beta = activation_params.get("beta", 1.0)
        # 运行时分发到对应内核
```

### 5. 扩展 CUDA 内核
**修改文件**: `sgl-kernel/csrc/elementwise/activation.cu`

```cpp
// 添加参数化 swish 函数
template <typename T>
__device__ __forceinline__ T swish(const T& x, float beta = 1.0f) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-beta * f32_val)));
}

// 添加灵活分发函数
template <typename T>
void flexible_act_and_mul(
    T* out,
    const T* input,
    int activation_type,
    float param,
    int num_tokens
) {
    switch(activation_type) {
        case 0: silu_and_mul(out, input, num_tokens); break;
        case 2: swish_and_mul(out, input, num_tokens, param); break;
        // ...
    }
}
```

## 📝 配置使用方法

### 1. 通过配置文件
```json
{
  "model_type": "deepseek_v3",
  "hidden_act": "swish",
  "activation_params": {
    "beta": 1.5
  }
}
```

### 2. 通过命令行参数
```bash
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --json-model-override-args '{"hidden_act": "swish", "activation_params": {"beta": 1.5}}'
```

### 3. 编程接口
```python
from sglang.srt.layers.activation_registry import ActivationRegistry

# 获取激活函数
activation = ActivationRegistry.get_activation("swish", {"beta": 1.3})
result = activation(input_tensor)

# 获取 GLU 风格激活函数
fused_activation = ActivationRegistry.get_fused_activation("swish", {"beta": 1.3})
result = fused_activation(gate_up_tensor)
```

## ✅ 如何验证实现成功

### 1. 检查日志输出
启动 SGLang 服务器时，应该看到如下日志：

```bash
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
    --json-model-override-args '{"hidden_act": "swish", "activation_params": {"beta": 1.5}}'

# 期望看到的日志
[2025-08-10 11:41:14] Parsed activation config: swish with params {'beta': 1.5}
```

**✅ 如果看到这行日志，说明激活函数配置解析成功！**

### 2. 编程验证
```python
# 测试脚本
import torch
from sglang.srt.layers.activation_registry import ActivationRegistry

# 测试基本功能
print("支持的激活函数:", ActivationRegistry.list_supported())

# 测试激活函数创建
swish = ActivationRegistry.get_activation("swish", {"beta": 1.5})
print("Swish 激活函数创建成功, beta =", swish.beta)

# 测试数值计算
x = torch.randn(10, 512)
result = swish(x)
print("激活函数计算成功, 输出形状:", result.shape)
```

### 3. 检查支持的激活函数
```python
from sglang.srt.layers.activation_registry import ActivationRegistry
supported = ActivationRegistry.list_supported()
print("支持的激活函数:", supported)

# 期望输出类似于:
# ['silu', 'gelu', 'swish', 'relu', 'leaky_relu', 'gelu_tanh', 'swiglu', 'geglu']
```

### 4. 测试配置解析
```python
from sglang.srt.layers.activation_registry import parse_activation_config

# 测试不同配置格式
configs = [
    {"hidden_act": "silu"},
    {"hidden_act": "swish", "activation_params": {"beta": 1.5}},
    {"activation_function": "gelu"},
    {},  # 空配置应该默认为 silu
]

for config in configs:
    name, params = parse_activation_config(config)
    print(f"配置 {config} -> 激活函数: {name}, 参数: {params}")
```

### 5. 验证模型加载
如果模型成功加载并且没有激活函数相关的错误，说明集成成功。特别是看到：

1. **配置解析日志**: `Parsed activation config: xxx`
2. **模型加载成功**: 没有 `Unsupported activation` 错误
3. **内核编译成功**: 没有 CUDA 编译错误

## 🚨 故障排除

### 1. 如果看到 "Unsupported activation function" 错误
```python
# 检查是否正确导入
from sglang.srt.layers.activation_registry import ActivationRegistry
print("注册的激活函数:", ActivationRegistry.list_supported())
```

### 2. 如果配置解析失败
检查配置格式是否正确：
```json
{
  "hidden_act": "swish",  // 确保键名正确
  "activation_params": {"beta": 1.5}  // 确保参数格式正确
}
```

### 3. 如果 CUDA 内核编译失败
```bash
# 重新编译内核
cd sgl-kernel
python setup.py build_ext --inplace
```

## 📋 实现检查清单

- [x] ✅ 激活函数注册系统创建完成
- [x] ✅ ModelConfig 支持激活函数配置解析  
- [x] ✅ Triton MoE 层支持动态激活函数
- [x] ✅ Cutlass MoE 层支持动态激活函数
- [x] ✅ CUDA 内核支持参数化激活函数
- [x] ✅ 配置兼容性支持（多种键名格式）
- [x] ✅ 向后兼容性保证

**验证成功的标志**: 启动 SGLang 服务器时看到 `Parsed activation config` 日志，并且模型能够正常加载运行。