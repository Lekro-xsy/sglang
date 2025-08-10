# SGLang MoE 激活函数灵活化支持

## 🎯 项目概述

本项目为 SGLang 的 MoE（专家混合模型）系统添加了**灵活的激活函数支持**，解决了之前硬编码激活函数的限制。现在可以通过配置文件灵活控制激活函数的选择和参数。

### 🌟 主要特性

- ✅ **多种激活函数支持**: SiLU, GELU, Swish, ReLU, LeakyReLU, GELU-Tanh
- ✅ **参数化激活**: 支持 Swish(β), LeakyReLU(α) 等带参数的激活函数
- ✅ **配置驱动**: 通过 `config.json` 或命令行参数控制
- ✅ **多后端优化**: Triton (constexpr) 和 Cutlass (launch-time) 
- ✅ **完全向后兼容**: 现有模型无需任何修改
- ✅ **高性能**: 针对不同后端优化，无性能损失

## 🏗️ 架构设计

```
SGLang MoE 激活函数架构
├── 配置层 (Config Layer)
│   ├── ModelConfig: 解析配置文件
│   └── ActivationRegistry: 统一注册管理
├── MoE 层 (MoE Layer)  
│   ├── Triton MoE: constexpr 专门化
│   └── Cutlass MoE: launch-time 分发
└── 内核层 (Kernel Layer)
    ├── Triton 内核: 编译时优化
    └── CUDA 内核: 运行时分发
```

## 📦 支持的模型

### MoE 模型系列 (Mixture of Experts)
- **DeepSeek v3/R1** (671B MoE)
- **Qwen3 MoE / Qwen2.5/2 MoE**
- **Llama 4 MoE** 变体
- **Mixtral** (Mistral MoE)
- **DBRX** (132B MoE)
- **OLMoE** (AllenAI)
- **XVERSE-MoE-A36B**
- **Phi-3.5-MoE**
- **ERNIE-4.5 MoE**
- **Granite 3.0 MoE**

### 标准模型系列 (Dense Models)
- **Llama** (2/3.x/4)
- **Mistral** 7B/NeMo/Small3
- **Gemma** (v1/v2/v3)
- **GLM-4 / ChatGLM**
- **InternLM 2**
- **Baichuan 2**
- **MiniCPM**
- **所有其他现有模型**

## 🚀 快速开始

### 1. 安装和设置

确保你已经安装了 SGLang：

```bash
# 克隆仓库
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 安装依赖
pip install -e ".[all]"
```

### 2. 基本用法

#### 方法 1: 通过配置文件

创建或修改模型的 `config.json`：

```json
{
  "model_type": "deepseek_v3",
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_attention_heads": 32,
  "hidden_act": "swish",
  "activation_params": {
    "beta": 1.2
  }
}
```

#### 方法 2: 通过命令行参数

```bash
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --json-model-override-args '{"hidden_act": "swish", "activation_params": {"beta": 1.5}}'
```

#### 方法 3: 通过环境变量

```bash
export SGLANG_ACTIVATION_FUNCTION="gelu"
export SGLANG_ACTIVATION_PARAMS='{"approximate": "tanh"}'
```

### 3. 编程接口

```python
from sglang.srt.layers.activation_registry import ActivationRegistry

# 获取激活函数实例
activation = ActivationRegistry.get_activation("swish", {"beta": 1.3})
result = activation(input_tensor)

# 获取 GLU 风格激活函数
fused_activation = ActivationRegistry.get_fused_activation("swish", {"beta": 1.3})
result = fused_activation(gate_up_tensor)  # 处理门控输入

# 检查支持的激活函数
supported = ActivationRegistry.list_supported()
print(f"Supported activations: {supported}")
```

## 🧪 验证和测试

### 1. 运行完整测试套件

```bash
# 进入项目根目录
cd /path/to/sglang

# 运行激活函数测试
python -m pytest test_activation_functions.py -v

# 运行特定测试类
python -m pytest test_activation_functions.py::TestActivationRegistry -v

# 运行性能测试
python -m pytest test_activation_functions.py::TestParameterKernel -v
```

### 2. 单元测试验证

#### 测试激活函数数值正确性

```python
# test_numerical_accuracy.py
import torch
import torch.nn.functional as F
from sglang.srt.layers.activation_registry import ActivationRegistry

def test_silu_accuracy():
    """验证 SiLU 激活函数数值正确性"""
    x = torch.randn(1024, 4096)
    
    # 参考实现
    expected = F.silu(x)
    
    # 我们的实现
    activation = ActivationRegistry.get_activation("silu")
    result = activation(x)
    
    # 验证数值误差 < 1e-5
    assert torch.allclose(result, expected, rtol=1e-5)
    print("✅ SiLU accuracy test passed")

def test_swish_with_beta():
    """验证带参数的 Swish 激活函数"""
    x = torch.randn(1024, 4096)
    beta = 1.5
    
    # 参考实现
    expected = x * torch.sigmoid(beta * x)
    
    # 我们的实现
    activation = ActivationRegistry.get_activation("swish", {"beta": beta})
    result = activation(x)
    
    # 验证数值误差 < 1e-5
    assert torch.allclose(result, expected, rtol=1e-5)
    print("✅ Swish with beta accuracy test passed")

if __name__ == "__main__":
    test_silu_accuracy()
    test_swish_with_beta()
```

### 3. MoE 层集成测试

#### 测试 Triton MoE 层

```python
# test_triton_moe.py
import torch
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput

def test_triton_moe_with_swish():
    """测试 Triton MoE 层使用 Swish 激活函数"""
    
    # 模拟配置
    num_experts = 8
    hidden_size = 4096
    intermediate_size = 11008
    top_k = 2
    
    # 创建 MoE 层
    moe_layer = FusedMoE(
        num_experts=num_experts,
        hidden_size=hidden_size, 
        intermediate_size=intermediate_size,
        layer_id=0,
        top_k=top_k,
        activation="swish",  # 使用 Swish 激活函数
        activation_alpha=1.2  # 设置 beta 参数
    )
    
    # 创建测试数据
    batch_size = 32
    hidden_states = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="cuda")
    
    # 模拟 TopK 输出
    topk_weights = torch.rand(batch_size, top_k, device="cuda")
    topk_ids = torch.randint(0, num_experts, (batch_size, top_k), device="cuda")
    topk_output = StandardTopKOutput(topk_weights, topk_ids)
    
    # 前向传播
    output = moe_layer(hidden_states, topk_output)
    
    # 验证输出形状
    assert output.shape == (batch_size, hidden_size)
    print("✅ Triton MoE with Swish test passed")

if __name__ == "__main__":
    test_triton_moe_with_swish()
```

### 4. 端到端模型测试

#### 测试 DeepSeek 模型加载

```python
# test_deepseek_model.py
import torch
from transformers import AutoConfig
from sglang.srt.models.deepseek import DeepseekMoE

def test_deepseek_with_custom_activation():
    """测试 DeepSeek 模型使用自定义激活函数"""
    
    # 创建配置，指定 Swish 激活函数
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat")
    config.hidden_act = "swish"
    config.activation_params = {"beta": 1.3}
    
    # 创建 MoE 层
    moe = DeepseekMoE(config)
    
    # 验证激活函数配置
    for expert in moe.experts:
        assert expert.act_fn.__class__.__name__ in ["SiluAndMul", "SwishAndMul"]
    
    print("✅ DeepSeek model with custom activation test passed")

if __name__ == "__main__":
    test_deepseek_with_custom_activation()
```

### 5. 性能基准测试

#### 性能回归测试

```bash
# benchmark_activation_performance.py
python benchmark/benchmark_moe_kernels.py \
    --activation silu --precision bf16 --batch_size 32 \
    --num_experts 8 --hidden_size 4096 > baseline_silu.txt

python benchmark/benchmark_moe_kernels.py \
    --activation swish --activation_params '{"beta": 1.0}' \
    --precision bf16 --batch_size 32 --num_experts 8 --hidden_size 4096 > result_swish.txt

# 比较性能，确保没有显著退化 (< 5%)
python scripts/compare_benchmark.py baseline_silu.txt result_swish.txt
```

#### 内存使用测试

```python
# test_memory_usage.py
import torch
import tracemalloc
from sglang.srt.layers.activation_registry import ActivationRegistry

def test_memory_usage():
    """测试激活函数内存使用"""
    
    tracemalloc.start()
    
    # 测试不同激活函数的内存使用
    x = torch.randn(1024, 4096, device="cuda")
    
    for activation_name in ["silu", "gelu", "swish"]:
        snapshot_before = tracemalloc.take_snapshot()
        
        activation = ActivationRegistry.get_activation(activation_name, {"beta": 1.2})
        result = activation(x)
        
        snapshot_after = tracemalloc.take_snapshot()
        
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        memory_diff = sum(stat.size_diff for stat in top_stats)
        
        print(f"✅ {activation_name}: Memory diff = {memory_diff} bytes")
        
        # 验证没有内存泄漏
        assert memory_diff < 1024 * 1024  # < 1MB

if __name__ == "__main__":
    test_memory_usage()
```

### 6. 兼容性测试

#### 向后兼容测试

```python
# test_backward_compatibility.py
def test_legacy_models():
    """测试现有模型的向后兼容性"""
    
    # 测试没有激活函数配置的模型
    config_without_activation = {}
    from sglang.srt.layers.activation_registry import parse_activation_config
    
    name, params = parse_activation_config(config_without_activation)
    assert name == "silu"  # 默认值
    assert params == {}
    print("✅ Legacy model compatibility test passed")

def test_multiple_config_formats():
    """测试多种配置格式的兼容性"""
    
    test_cases = [
        {"hidden_act": "gelu"},
        {"activation_function": "swish", "activation_params": {"beta": 1.5}},
        {"hidden_activation": "silu"},
        {"activation": "relu"}
    ]
    
    from sglang.srt.layers.activation_registry import parse_activation_config
    
    for config in test_cases:
        name, params = parse_activation_config(config)
        assert name in ["gelu", "swish", "silu", "relu"]
        print(f"✅ Config format {config} parsed successfully")

if __name__ == "__main__":
    test_legacy_models()
    test_multiple_config_formats()
```

## 🔧 故障排除

### 常见问题与解决方案

#### 1. 激活函数不支持错误

```
ValueError: Unsupported activation function 'custom_activation'
```

**解决方案**：
```python
# 检查支持的激活函数
from sglang.srt.layers.activation_registry import ActivationRegistry
print("Supported activations:", ActivationRegistry.list_supported())

# 或者添加自定义激活函数
class CustomActivation(ParameterizedActivation):
    def __init__(self, params=None):
        super().__init__(ActivationType.CUSTOM, params)
        
    def forward(self, x):
        return your_custom_activation(x)

ActivationRegistry._ACTIVATION_CLASSES["custom_activation"] = CustomActivation
```

#### 2. 配置解析失败

```
Warning: Failed to parse activation config: KeyError: 'hidden_act'
```

**解决方案**：
```python
# 检查配置文件格式
config = {
    "hidden_act": "swish",  # 确保键名正确
    "activation_params": {"beta": 1.2}  # 参数格式正确
}
```

#### 3. CUDA 内核编译错误

```
Error: CUDA kernel compilation failed
```

**解决方案**：
```bash
# 重新编译 CUDA 内核
cd sgl-kernel
python setup.py build_ext --inplace

# 检查 CUDA 版本兼容性
nvidia-smi
nvcc --version
```

#### 4. 性能下降问题

**诊断步骤**：
```python
# 1. 检查激活函数类型
print(f"Activation type: {moe_layer.activation_type}")
print(f"Activation params: {moe_layer.activation_params}")

# 2. 验证是否使用了优化路径
# 对于 Swish，beta=1.0 应该自动回退到 SiLU
if activation_params.get("beta") == 1.0:
    print("Using SiLU optimization for Swish(beta=1.0)")

# 3. 检查内核使用情况
print(f"Using Triton kernels: {moe_layer.use_triton_kernels}")
```

## 📚 API 参考

### ActivationRegistry

```python
class ActivationRegistry:
    @classmethod
    def get_activation(cls, name: str, params: Optional[Dict] = None) -> ParameterizedActivation
    
    @classmethod
    def get_fused_activation(cls, name: str, params: Optional[Dict] = None) -> Union[SiluAndMul, GeluAndMul, ...]
    
    @classmethod
    def is_supported(cls, name: str) -> bool
    
    @classmethod
    def list_supported(cls) -> List[str]
```

### 支持的激活函数

| 激活函数 | 参数 | 描述 |
|---------|------|------|
| `silu` | 无 | x * sigmoid(x) |
| `gelu` | `approximate`: "none"/"tanh" | GELU 激活函数 |
| `swish` | `beta`: float | x * sigmoid(beta * x) |
| `relu` | 无 | max(0, x) |
| `leaky_relu` | `alpha`: float | max(alpha * x, x) |
| `gelu_tanh` | 无 | GELU 的 tanh 近似 |

### 配置选项

```python
# 配置文件中的键名（按优先级）
activation_keys = [
    "hidden_act",           # HuggingFace 标准
    "activation_function",  # 替代键
    "hidden_activation",    # GLM 风格
    "activation",          # 简单键
]

param_keys = [
    "activation_params",    # 标准参数键
    "activation_kwargs",    # 替代参数键
    "act_params",          # 简短参数键
]
```

## 🤝 贡献指南

### 添加新的激活函数

1. **Python 层实现**：
```python
# 在 activation_registry.py 中
class NewActivation(ParameterizedActivation):
    def __init__(self, params=None):
        super().__init__(ActivationType.NEW, params)
        self.param1 = params.get("param1", 1.0) if params else 1.0
        
    def forward(self, x):
        return new_activation_function(x, self.param1)
    
    def get_kernel_params(self):
        return {"param1": self.param1}
```

2. **注册激活函数**：
```python
# 添加到类型枚举
class ActivationType(IntEnum):
    # ...
    NEW = 6

# 添加到注册表
ActivationRegistry._ACTIVATION_CLASSES["new_activation"] = NewActivation
ActivationRegistry._TYPE_TO_NAME[ActivationType.NEW] = "new_activation"
```

3. **CUDA 内核实现**：
```cpp
// 在 activation.cu 中
template <typename T>
__device__ __forceinline__ T new_activation(const T& x, float param1) {
    // 实现新的激活函数
    return custom_compute(x, param1);
}
```

4. **添加测试**：
```python
def test_new_activation():
    x = torch.randn(1024, 4096)
    activation = ActivationRegistry.get_activation("new_activation", {"param1": 2.0})
    result = activation(x)
    # 验证结果
    assert result.shape == x.shape
```

## 📄 许可证

本项目遵循 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢 SGLang 团队和社区的支持，以及所有贡献者的努力！

---

**快速链接**：
- [项目主页](https://github.com/sgl-project/sglang)
- [文档](https://sgl-project.github.io/sglang/)
- [问题反馈](https://github.com/sgl-project/sglang/issues)
- [讨论区](https://github.com/sgl-project/sglang/discussions)