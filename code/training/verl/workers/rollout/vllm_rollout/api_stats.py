#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 调用统计器
用于统计模拟器 API 调用的总量、token 使用情况等
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor


@dataclass
class APIStats:
    """API 调用统计信息"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0  # 输入 + 输出
    estimated_cost: float = 0.0  # 预估成本（元）
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # 按场景类型统计
    scene_stats: Dict[str, 'APIStats'] = field(default_factory=dict)

    # 错误统计
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def add_call(self, scene_tag: str, input_tokens: int, output_tokens: int,
                 success: bool = True, error_msg: str = None, cost: float = 0.0,
                 record_scene: bool = True):
        """记录一次 API 调用"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if error_msg:
                self.errors.append({
                    'timestamp': time.time(),
                    'scene_tag': scene_tag,
                    'error': error_msg,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                })

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)
        self.estimated_cost += cost

        # 更新场景统计
        if record_scene:
            if scene_tag not in self.scene_stats:
                self.scene_stats[scene_tag] = APIStats()
            self.scene_stats[scene_tag].add_call(
                scene_tag,
                input_tokens,
                output_tokens,
                success,
                error_msg,
                cost,
                record_scene=False,
            )

    def finish(self):
        """完成统计"""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """获取统计时长（秒）"""
        end_time = self.end_time or time.time()
        return end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.successful_calls / max(self.total_calls, 1),
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_tokens,
            'estimated_cost': round(self.estimated_cost, 4),
            'duration_seconds': round(self.get_duration(), 2),
            'calls_per_second': round(self.total_calls / max(self.get_duration(), 1), 2),
            'tokens_per_second': round(self.total_tokens / max(self.get_duration(), 1), 2),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'scene_stats': {k: v.to_dict() for k, v in self.scene_stats.items()},
            'error_count': len(self.errors),
            'recent_errors': self.errors[-10:] if self.errors else []  # 只保留最近10个错误
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIStats":
        """从字典构造 APIStats 实例（用于合并/加载）"""
        stats = cls(
            total_calls=int(data.get('total_calls', 0) or 0),
            successful_calls=int(data.get('successful_calls', 0) or 0),
            failed_calls=int(data.get('failed_calls', 0) or 0),
            total_input_tokens=int(data.get('total_input_tokens', 0) or 0),
            total_output_tokens=int(data.get('total_output_tokens', 0) or 0),
            total_tokens=int(data.get('total_tokens', 0) or 0),
            estimated_cost=float(data.get('estimated_cost', 0.0) or 0.0),
            start_time=float(data.get('start_time', time.time()) or time.time()),
            end_time=data.get('end_time'),
        )
        # scene stats
        for scene, scene_data in (data.get('scene_stats') or {}).items():
            stats.scene_stats[scene] = cls.from_dict(scene_data)

        # recent errors
        stats.errors = list(data.get('recent_errors') or [])
        return stats

    def save_to_file(self, filepath: str):
        """保存统计结果到文件"""
        dir_path = os.path.dirname(filepath)
        if dir_path:  # 只有当有目录路径时才创建目录
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def print_summary(self):
        """打印统计摘要"""
        duration = self.get_duration()
        print("\n=== API 调用统计摘要 ===")
        print(f"总调用次数: {self.total_calls}")
        print(f"成功次数: {self.successful_calls} ({self.successful_calls/max(self.total_calls,1)*100:.1f}%)")
        print(f"失败次数: {self.failed_calls}")
        print(f"总输入 Token: {self.total_input_tokens:,}")
        print(f"总输出 Token: {self.total_output_tokens:,}")
        print(f"总 Token: {self.total_tokens:,}")
        print(f"预估成本: ¥{self.estimated_cost:.4f}")
        print(f"统计时长: {duration:.2f} 秒")
        print(f"调用频率: {self.total_calls/max(duration,1):.2f} 次/秒")
        print(f"Token 频率: {self.total_tokens/max(duration,1):.2f} Token/秒")

        if self.scene_stats:
            print("\n按场景统计:")
            for scene, stats in self.scene_stats.items():
                success_rate = stats.successful_calls / max(stats.total_calls, 1) * 100
                print(f"  {scene}: {stats.total_calls} 次 ({success_rate:.1f}% 成功), "
                      f"{stats.total_tokens:,} Token, ¥{stats.estimated_cost:.4f}")

        if self.errors:
            print(f"\n错误统计: {len(self.errors)} 个错误")
            print("最近错误:")
            for i, error in enumerate(self.errors[-3:]):
                print(f"  {i+1}. {error.get('error', 'Unknown error')[:100]}...")


class APIStatsManager:
    """API 统计管理器（线程安全）"""

    def __init__(self):
        self._stats = APIStats()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="api_stats")

    def record_call(self, scene_tag: str, input_tokens: int, output_tokens: int,
                   success: bool = True, error_msg: str = None, model: str = "qwen3-max"):
        """异步记录一次 API 调用"""
        # 简单估算成本（基于通义千问价格，实际可能不同）
        cost_per_token = 0.0032 if "qwen3-max" in model else 0.0001  # 每千token价格
        estimated_cost = (input_tokens + output_tokens) * cost_per_token / 1000

        def _record():
            with self._lock:
                self._stats.add_call(scene_tag, input_tokens, output_tokens, success, error_msg, estimated_cost)

        self._executor.submit(_record)

    def get_stats(self) -> APIStats:
        """获取当前统计信息"""
        with self._lock:
            return self._stats

    def save_stats(self, filepath: str):
        """保存统计信息"""
        with self._lock:
            self._stats.finish()
            self._stats.save_to_file(filepath)

    def print_summary(self):
        """打印统计摘要"""
        with self._lock:
            self._stats.print_summary()

    def shutdown(self):
        """关闭统计管理器"""
        self._executor.shutdown(wait=True)


# 全局统计实例
_global_stats_manager = None

def get_global_stats_manager() -> APIStatsManager:
    """获取全局统计管理器"""
    global _global_stats_manager
    if _global_stats_manager is None:
        _global_stats_manager = APIStatsManager()
    return _global_stats_manager

def init_global_stats():
    """初始化全局统计"""
    global _global_stats_manager
    if _global_stats_manager is not None:
        _global_stats_manager.shutdown()
    _global_stats_manager = APIStatsManager()

def save_global_stats(filepath: str):
    """保存全局统计"""
    manager = get_global_stats_manager()
    manager.save_stats(filepath)

def print_global_stats():
    """打印全局统计"""
    manager = get_global_stats_manager()
    manager.print_summary()


# 便捷函数
def record_api_call(scene_tag: str, input_tokens: int, output_tokens: int,
                   success: bool = True, error_msg: str = None, model: str = "qwen3-max"):
    """记录一次 API 调用（便捷函数）"""
    manager = get_global_stats_manager()
    manager.record_call(scene_tag, input_tokens, output_tokens, success, error_msg, model)


if __name__ == "__main__":
    # 测试代码
    init_global_stats()

    # 模拟一些 API 调用
    record_api_call("support", 1000, 500, success=True, model="qwen3-max")
    record_api_call("defense", 1200, 400, success=True, model="qwen3-max")
    record_api_call("repair", 800, 600, success=False, error_msg="Timeout", model="qwen3-max")
    record_api_call("charm", 1500, 300, success=True, model="qwen3-max")

    # 等待统计完成
    time.sleep(0.1)

    # 打印和保存
    print_global_stats()
    save_global_stats("api_stats_test.json")
    print("统计已保存到 api_stats_test.json")
