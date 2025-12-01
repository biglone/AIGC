#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型量化工具 - INT8/INT4量化实现
==============================

项目目标：
- 理解量化原理和实现细节
- 掌握INT8/INT4量化技术
- 实现实用的量化工具

量化技术：
1. INT8 对称量化
2. INT8 非对称量化
3. INT4 分组量化
4. 精度评估和对比

作者：面向C++工程师的AIGC学习
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from dataclasses import dataclass
import time


# ============================================================
# 第一部分：INT8量化
# ============================================================

class INT8Quantizer:
    """
    INT8量化器

    对称量化：range = [-127, 127]
    量化公式：q = round(x / scale)
    scale = max(abs(x)) / 127

    类比C++：类似于将float转换为int8_t，需要记录scale因子
    """

    def __init__(self, symmetric: bool = True):
        """
        Args:
            symmetric: True=对称量化[-127,127], False=非对称量化[0,255]
        """
        self.symmetric = symmetric
        self.scale = None
        self.zero_point = None  # 非对称量化使用

    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        量化FP32权重到INT8

        Args:
            x: FP32权重矩阵

        Returns:
            (quantized_weights, metadata)
            quantized_weights: INT8量化后的权重
            metadata: {'scale': float, 'zero_point': int}
        """
        if self.symmetric:
            return self._quantize_symmetric(x)
        else:
            return self._quantize_asymmetric(x)

    def _quantize_symmetric(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        对称量化：[-127, 127]

        优点：实现简单，只需一个scale参数
        缺点：如果数据分布不对称，会有精度损失
        """
        # 计算scale
        max_val = np.max(np.abs(x))
        if max_val == 0:
            self.scale = 1.0
        else:
            self.scale = max_val / 127.0

        # 量化
        x_quantized = np.round(x / self.scale)
        x_quantized = np.clip(x_quantized, -127, 127).astype(np.int8)

        metadata = {
            'scale': self.scale,
            'zero_point': 0,
            'symmetric': True
        }

        return x_quantized, metadata

    def _quantize_asymmetric(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        非对称量化：[0, 255]

        优点：能更好处理非对称分布
        缺点：需要额外的zero_point参数
        """
        # 计算range
        x_min = np.min(x)
        x_max = np.max(x)

        # 计算scale和zero_point
        if x_max - x_min == 0:
            self.scale = 1.0
            self.zero_point = 0
        else:
            self.scale = (x_max - x_min) / 255.0
            self.zero_point = int(-x_min / self.scale)

        # 量化
        x_quantized = np.round(x / self.scale + self.zero_point)
        x_quantized = np.clip(x_quantized, 0, 255).astype(np.uint8)

        metadata = {
            'scale': self.scale,
            'zero_point': self.zero_point,
            'symmetric': False
        }

        return x_quantized, metadata

    def dequantize(self, x_quantized: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        反量化：INT8 -> FP32

        Args:
            x_quantized: 量化后的权重
            metadata: 量化元数据

        Returns:
            FP32权重（近似恢复）
        """
        scale = metadata['scale']
        zero_point = metadata.get('zero_point', 0)

        # 反量化
        x = (x_quantized.astype(np.float32) - zero_point) * scale

        return x


# ============================================================
# 第二部分：INT4量化（分组量化）
# ============================================================

class INT4GroupQuantizer:
    """
    INT4分组量化

    核心思想：
    1. 将权重矩阵分成多个组（如128个元素一组）
    2. 每组独立计算scale
    3. 每组量化到[-7, 7]（4bit有符号）

    优势：
    - 更细粒度的scale，精度更高
    - 内存节省75%（相比FP32）
    - INT4计算速度快

    类比C++：类似于将数组分块，每块独立编码
    """

    def __init__(self, group_size: int = 128):
        """
        Args:
            group_size: 每组的元素个数
        """
        self.group_size = group_size

    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        INT4分组量化

        Args:
            x: FP32权重矩阵 (可以是任意形状)

        Returns:
            (quantized_weights, metadata)
        """
        original_shape = x.shape
        x_flat = x.flatten()
        n = len(x_flat)

        # 计算需要padding的元素数
        n_groups = (n + self.group_size - 1) // self.group_size
        padded_size = n_groups * self.group_size
        padding_size = padded_size - n

        if padding_size > 0:
            x_flat = np.pad(x_flat, (0, padding_size), mode='constant')

        # 分组
        x_grouped = x_flat.reshape(n_groups, self.group_size)

        # 每组独立量化
        quantized_groups = []
        scales = []

        for group in x_grouped:
            # 计算组内scale
            max_val = np.max(np.abs(group))
            if max_val == 0:
                scale = 1.0
            else:
                scale = max_val / 7.0  # INT4: [-7, 7]

            scales.append(scale)

            # 量化到INT4 [-7, 7]
            group_quantized = np.round(group / scale)
            group_quantized = np.clip(group_quantized, -7, 7).astype(np.int8)
            quantized_groups.append(group_quantized)

        # 合并
        x_quantized = np.concatenate(quantized_groups)

        # 移除padding
        if padding_size > 0:
            x_quantized = x_quantized[:-padding_size]

        metadata = {
            'scales': np.array(scales, dtype=np.float32),
            'group_size': self.group_size,
            'original_shape': original_shape,
            'n_groups': n_groups
        }

        return x_quantized, metadata

    def dequantize(self, x_quantized: np.ndarray, metadata: Dict) -> np.ndarray:
        """反量化：INT4 -> FP32"""
        scales = metadata['scales']
        group_size = metadata['group_size']
        original_shape = metadata['original_shape']
        n_groups = metadata['n_groups']

        # Padding到完整组
        n = len(x_quantized)
        padded_size = n_groups * group_size
        padding_size = padded_size - n

        if padding_size > 0:
            x_quantized = np.pad(x_quantized, (0, padding_size), mode='constant')

        # 分组
        x_grouped = x_quantized.reshape(n_groups, group_size)

        # 每组独立反量化
        dequantized_groups = []
        for i, (group, scale) in enumerate(zip(x_grouped, scales)):
            group_dequant = group.astype(np.float32) * scale
            dequantized_groups.append(group_dequant)

        # 合并
        x_dequant = np.concatenate(dequantized_groups)

        # 移除padding
        if padding_size > 0:
            x_dequant = x_dequant[:-padding_size]

        # 恢复原始形状
        x_dequant = x_dequant.reshape(original_shape)

        return x_dequant


# ============================================================
# 第三部分：量化精度评估
# ============================================================

class QuantizationEvaluator:
    """量化精度评估器"""

    @staticmethod
    def evaluate(original: np.ndarray, reconstructed: np.ndarray) -> Dict:
        """
        评估量化精度

        指标：
        1. MSE (Mean Squared Error)
        2. RMSE (Root Mean Squared Error)
        3. MAE (Mean Absolute Error)
        4. Max Error (最大误差)
        5. SQNR (Signal-to-Quantization-Noise Ratio)
        """
        # MSE
        mse = np.mean((original - reconstructed) ** 2)

        # RMSE
        rmse = np.sqrt(mse)

        # MAE
        mae = np.mean(np.abs(original - reconstructed))

        # Max Error
        max_error = np.max(np.abs(original - reconstructed))

        # SQNR (dB)
        signal_power = np.mean(original ** 2)
        noise_power = mse
        if noise_power > 0:
            sqnr_db = 10 * np.log10(signal_power / noise_power)
        else:
            sqnr_db = float('inf')

        # 相对误差
        relative_error = mae / (np.mean(np.abs(original)) + 1e-10)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'sqnr_db': sqnr_db,
            'relative_error': relative_error
        }

    @staticmethod
    def visualize_comparison(original: np.ndarray, reconstructed: np.ndarray, title: str = ""):
        """可视化对比"""
        plt.figure(figsize=(15, 5))

        # 子图1：原始 vs 重建
        plt.subplot(1, 3, 1)
        plt.scatter(original.flatten(), reconstructed.flatten(), alpha=0.1, s=1)
        plt.plot([original.min(), original.max()], [original.min(), original.max()], 'r--', label='理想')
        plt.xlabel('原始值')
        plt.ylabel('重建值')
        plt.title('原始 vs 重建')
        plt.legend()
        plt.grid(True)

        # 子图2：误差分布
        plt.subplot(1, 3, 2)
        errors = (original - reconstructed).flatten()
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('误差')
        plt.ylabel('频数')
        plt.title(f'误差分布 (MAE={np.mean(np.abs(errors)):.6f})')
        plt.grid(True)

        # 子图3：原始和重建值的分布
        plt.subplot(1, 3, 3)
        plt.hist(original.flatten(), bins=50, alpha=0.5, label='原始', edgecolor='black')
        plt.hist(reconstructed.flatten(), bins=50, alpha=0.5, label='重建', edgecolor='black')
        plt.xlabel('值')
        plt.ylabel('频数')
        plt.title('值分布对比')
        plt.legend()
        plt.grid(True)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


# ============================================================
# 第四部分：实战测试
# ============================================================

def test_int8_quantization():
    """测试INT8量化"""
    print("=" * 60)
    print("测试1：INT8对称量化")
    print("=" * 60)

    # 创建模拟权重矩阵（正态分布）
    np.random.seed(42)
    weights = np.random.randn(1000, 1000).astype(np.float32) * 0.1

    print(f"原始权重: shape={weights.shape}, dtype={weights.dtype}")
    print(f"  数值范围: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  内存占用: {weights.nbytes / 1024 / 1024:.2f} MB")

    # INT8对称量化
    quantizer = INT8Quantizer(symmetric=True)
    start = time.time()
    weights_q, metadata = quantizer.quantize(weights)
    quant_time = time.time() - start

    print(f"\nINT8量化后:")
    print(f"  shape={weights_q.shape}, dtype={weights_q.dtype}")
    print(f"  数值范围: [{weights_q.min()}, {weights_q.max()}]")
    print(f"  scale: {metadata['scale']:.6f}")
    print(f"  内存占用: {weights_q.nbytes / 1024 / 1024:.2f} MB")
    print(f"  内存节省: {(1 - weights_q.nbytes / weights.nbytes) * 100:.1f}%")
    print(f"  量化耗时: {quant_time * 1000:.2f} ms")

    # 反量化
    start = time.time()
    weights_dq = quantizer.dequantize(weights_q, metadata)
    dequant_time = time.time() - start

    # 评估精度
    metrics = QuantizationEvaluator.evaluate(weights, weights_dq)
    print(f"\n精度评估:")
    print(f"  MSE:          {metrics['mse']:.8f}")
    print(f"  RMSE:         {metrics['rmse']:.8f}")
    print(f"  MAE:          {metrics['mae']:.8f}")
    print(f"  最大误差:     {metrics['max_error']:.8f}")
    print(f"  SQNR:         {metrics['sqnr_db']:.2f} dB")
    print(f"  相对误差:     {metrics['relative_error'] * 100:.3f}%")
    print(f"  反量化耗时:   {dequant_time * 1000:.2f} ms")

    return weights, weights_dq


def test_int4_quantization():
    """测试INT4分组量化"""
    print("\n" + "=" * 60)
    print("测试2：INT4分组量化")
    print("=" * 60)

    # 创建模拟权重矩阵
    np.random.seed(42)
    weights = np.random.randn(1000, 1000).astype(np.float32) * 0.1

    print(f"原始权重: shape={weights.shape}, dtype={weights.dtype}")
    print(f"  内存占用: {weights.nbytes / 1024 / 1024:.2f} MB")

    # INT4分组量化
    quantizer = INT4GroupQuantizer(group_size=128)
    start = time.time()
    weights_q, metadata = quantizer.quantize(weights)
    quant_time = time.time() - start

    # 注意：INT4实际还是用int8存储，每个int8可以存2个INT4值
    # 这里为了简化演示，直接用int8存储单个INT4值
    actual_memory = weights_q.nbytes / 2  # 实际可以压缩一半

    print(f"\nINT4量化后:")
    print(f"  shape={weights_q.shape}, dtype={weights_q.dtype}")
    print(f"  group_size: {metadata['group_size']}")
    print(f"  n_groups: {metadata['n_groups']}")
    print(f"  scales: {len(metadata['scales'])} 个")
    print(f"  理论内存占用: {actual_memory / 1024 / 1024:.2f} MB (pack后)")
    print(f"  内存节省: {(1 - actual_memory / weights.nbytes) * 100:.1f}%")
    print(f"  量化耗时: {quant_time * 1000:.2f} ms")

    # 反量化
    start = time.time()
    weights_dq = quantizer.dequantize(weights_q, metadata)
    dequant_time = time.time() - start

    # 评估精度
    metrics = QuantizationEvaluator.evaluate(weights, weights_dq)
    print(f"\n精度评估:")
    print(f"  MSE:          {metrics['mse']:.8f}")
    print(f"  RMSE:         {metrics['rmse']:.8f}")
    print(f"  MAE:          {metrics['mae']:.8f}")
    print(f"  最大误差:     {metrics['max_error']:.8f}")
    print(f"  SQNR:         {metrics['sqnr_db']:.2f} dB")
    print(f"  相对误差:     {metrics['relative_error'] * 100:.3f}%")
    print(f"  反量化耗时:   {dequant_time * 1000:.2f} ms")

    return weights, weights_dq


def compare_quantization_methods():
    """对比不同量化方法"""
    print("\n" + "=" * 60)
    print("测试3：量化方法对比")
    print("=" * 60)

    # 创建测试数据
    np.random.seed(42)
    weights = np.random.randn(1000, 1000).astype(np.float32) * 0.1

    results = {}

    # FP32 baseline
    results['FP32'] = {
        'memory_mb': weights.nbytes / 1024 / 1024,
        'memory_ratio': 1.0,
        'mae': 0,
        'sqnr_db': float('inf')
    }

    # INT8对称量化
    q8_sym = INT8Quantizer(symmetric=True)
    w8_q, meta8 = q8_sym.quantize(weights)
    w8_dq = q8_sym.dequantize(w8_q, meta8)
    metrics8 = QuantizationEvaluator.evaluate(weights, w8_dq)
    results['INT8-Sym'] = {
        'memory_mb': w8_q.nbytes / 1024 / 1024,
        'memory_ratio': w8_q.nbytes / weights.nbytes,
        'mae': metrics8['mae'],
        'sqnr_db': metrics8['sqnr_db']
    }

    # INT8非对称量化
    q8_asym = INT8Quantizer(symmetric=False)
    w8a_q, meta8a = q8_asym.quantize(weights)
    w8a_dq = q8_asym.dequantize(w8a_q, meta8a)
    metrics8a = QuantizationEvaluator.evaluate(weights, w8a_dq)
    results['INT8-Asym'] = {
        'memory_mb': w8a_q.nbytes / 1024 / 1024,
        'memory_ratio': w8a_q.nbytes / weights.nbytes,
        'mae': metrics8a['mae'],
        'sqnr_db': metrics8a['sqnr_db']
    }

    # INT4分组量化
    q4 = INT4GroupQuantizer(group_size=128)
    w4_q, meta4 = q4.quantize(weights)
    w4_dq = q4.dequantize(w4_q, meta4)
    metrics4 = QuantizationEvaluator.evaluate(weights, w4_dq)
    results['INT4-Group'] = {
        'memory_mb': w4_q.nbytes / 2 / 1024 / 1024,  # pack后
        'memory_ratio': (w4_q.nbytes / 2) / weights.nbytes,
        'mae': metrics4['mae'],
        'sqnr_db': metrics4['sqnr_db']
    }

    # 打印对比表格
    print("\n量化方法对比:")
    print(f"{'方法':<12} {'内存(MB)':<10} {'内存比例':<10} {'MAE':<12} {'SQNR(dB)':<10}")
    print("-" * 60)
    for method, result in results.items():
        print(f"{method:<12} {result['memory_mb']:<10.2f} {result['memory_ratio']:<10.1%} "
              f"{result['mae']:<12.6f} {result['sqnr_db']:<10.2f}")

    print("\n总结:")
    print("• INT8对称量化：内存节省75%，精度损失<1%，实现简单")
    print("• INT8非对称量化：内存节省75%，精度略好于对称量化")
    print("• INT4分组量化：内存节省87.5%，精度取决于group_size")
    print("• 实际生产环境推荐：INT8用于大模型推理，INT4用于资源受限场景")


def demo_quantization_workflow():
    """完整量化工作流演示"""
    print("\n" + "=" * 60)
    print("完整量化工作流演示")
    print("=" * 60)

    # 模拟一个小型神经网络的权重
    print("\n[步骤1] 创建模拟神经网络权重...")
    np.random.seed(42)
    layer_weights = {
        'layer1': np.random.randn(512, 512).astype(np.float32) * 0.1,
        'layer2': np.random.randn(512, 256).astype(np.float32) * 0.05,
        'layer3': np.random.randn(256, 128).astype(np.float32) * 0.02,
        'output': np.random.randn(128, 10).astype(np.float32) * 0.01
    }

    total_params = sum(w.size for w in layer_weights.values())
    total_memory = sum(w.nbytes for w in layer_weights.values()) / 1024 / 1024
    print(f"✓ 网络参数总数: {total_params:,}")
    print(f"✓ FP32内存占用: {total_memory:.2f} MB")

    # 量化所有层
    print("\n[步骤2] 量化所有层权重...")
    quantizer = INT8Quantizer(symmetric=True)
    quantized_weights = {}
    metadata_dict = {}

    for name, weights in layer_weights.items():
        w_q, meta = quantizer.quantize(weights)
        quantized_weights[name] = w_q
        metadata_dict[name] = meta
        print(f"  ✓ {name}: {weights.shape} -> INT8")

    quantized_memory = sum(w.nbytes for w in quantized_weights.values()) / 1024 / 1024
    print(f"✓ INT8内存占用: {quantized_memory:.2f} MB")
    print(f"✓ 内存节省: {(1 - quantized_memory / total_memory) * 100:.1f}%")

    # 评估整体精度
    print("\n[步骤3] 评估量化精度...")
    total_mae = 0
    for name, weights in layer_weights.items():
        w_q = quantized_weights[name]
        meta = metadata_dict[name]
        w_dq = quantizer.dequantize(w_q, meta)
        metrics = QuantizationEvaluator.evaluate(weights, w_dq)
        total_mae += metrics['mae']
        print(f"  {name}: MAE={metrics['mae']:.6f}, SQNR={metrics['sqnr_db']:.2f}dB")

    print(f"✓ 平均MAE: {total_mae / len(layer_weights):.6f}")

    print("\n[工作流完成]")
    print("实际部署时:")
    print("1. 保存量化后的权重（INT8数组）")
    print("2. 保存量化元数据（scales）")
    print("3. 推理时在运行时反量化或使用INT8 kernel")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════╗
║          模型量化工具 - INT8/INT4 实现                  ║
║                                                        ║
║  支持的量化方法:                                        ║
║    • INT8 对称量化                                      ║
║    • INT8 非对称量化                                    ║
║    • INT4 分组量化                                      ║
║                                                        ║
║  核心技术:                                             ║
║    • 量化/反量化算法                                    ║
║    • 精度评估                                          ║
║    • 内存优化                                          ║
║                                                        ║
║  适合场景:                                             ║
║    • 学习量化原理                                      ║
║    • 模型压缩                                          ║
║    • 面试项目展示                                      ║
╚════════════════════════════════════════════════════════╝
    """)

    # 测试INT8量化
    weights_orig, weights_int8 = test_int8_quantization()

    # 测试INT4量化
    _, weights_int4 = test_int4_quantization()

    # 对比不同方法
    compare_quantization_methods()

    # 完整工作流
    demo_quantization_workflow()

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)

    print("\n实际应用建议:")
    print("1. INT8量化适合大多数场景，精度损失<1%")
    print("2. INT4适合资源受限场景（如边缘设备）")
    print("3. 分组量化比全局量化精度更高")
    print("4. 实际生产推荐使用成熟框架：")
    print("   - PyTorch: torch.quantization")
    print("   - TensorRT: INT8 calibration")
    print("   - ONNX Runtime: quantization tools")
    print("\n5. C++实现可以使用SIMD指令优化量化/反量化过程")
