#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 统计工具函数
提供训练时使用的初始化和保存功能
"""

import os
import atexit
import glob
import json
from typing import Dict, List, Tuple
from .api_stats import init_global_stats, save_global_stats, print_global_stats, APIStats


def init_api_stats_for_training():
    """在训练开始时初始化 API 统计"""
    init_global_stats()
    print("API 统计已初始化，将记录所有模拟器 API 调用")


def _get_stats_file(output_dir: str, suffix: str = "training", per_process: bool = False) -> str:
    """构造统计文件路径"""
    filename = f"api_stats_{suffix}.json"
    if per_process:
        filename = f"api_stats_{suffix}_{os.getpid()}.json"
    return os.path.join(output_dir, filename)


def save_api_stats_on_exit(output_dir: str = "api_stats", per_process: bool = False) -> str:
    """在训练结束时保存 API 统计"""
    os.makedirs(output_dir, exist_ok=True)
    stats_file = _get_stats_file(output_dir, suffix="training", per_process=per_process)
    save_global_stats(stats_file)
    print(f"API 统计已保存到: {stats_file}")
    print_global_stats()
    return stats_file


def setup_api_stats_for_training(output_dir: str = "api_stats", per_process: bool = False):
    """设置训练时的 API 统计（初始化 + 自动保存）"""
    init_api_stats_for_training()

    # 注册退出时保存统计
    atexit.register(save_api_stats_on_exit, output_dir, per_process)


def _merge_stat_dict(target: Dict, src: Dict):
    """将单个统计 dict 合并到 target 中（就地更新）"""
    target['total_calls'] += int(src.get('total_calls', 0) or 0)
    target['successful_calls'] += int(src.get('successful_calls', 0) or 0)
    target['failed_calls'] += int(src.get('failed_calls', 0) or 0)
    target['total_input_tokens'] += int(src.get('total_input_tokens', 0) or 0)
    target['total_output_tokens'] += int(src.get('total_output_tokens', 0) or 0)
    target['total_tokens'] += int(src.get('total_tokens', 0) or 0)
    target['estimated_cost'] += float(src.get('estimated_cost', 0.0) or 0.0)

    start_time = src.get('start_time')
    end_time = src.get('end_time')
    if start_time is not None:
        target['start_time'] = start_time if target['start_time'] is None else min(target['start_time'], start_time)
    if end_time is not None:
        target['end_time'] = end_time if target['end_time'] is None else max(target['end_time'], end_time)

    target['recent_errors'].extend(src.get('recent_errors') or [])

    for scene, scene_stats in (src.get('scene_stats') or {}).items():
        if scene not in target['scene_stats']:
            target['scene_stats'][scene] = _empty_stat_dict()
        _merge_stat_dict(target['scene_stats'][scene], scene_stats)


def _empty_stat_dict() -> Dict:
    return {
        'total_calls': 0,
        'successful_calls': 0,
        'failed_calls': 0,
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'total_tokens': 0,
        'estimated_cost': 0.0,
        'start_time': None,
        'end_time': None,
        'scene_stats': {},
        'recent_errors': [],
    }


def _finalize_stat_dict(stat: Dict) -> Dict:
    """计算派生指标"""
    duration = 0.0
    if stat['start_time'] is not None and stat['end_time'] is not None:
        duration = max(0.0, float(stat['end_time']) - float(stat['start_time']))
    stat['duration_seconds'] = round(duration, 2)
    stat['success_rate'] = stat['successful_calls'] / max(stat['total_calls'], 1)
    stat['calls_per_second'] = round(stat['total_calls'] / max(duration, 1), 2) if duration else 0.0
    stat['tokens_per_second'] = round(stat['total_tokens'] / max(duration, 1), 2) if duration else 0.0
    stat['error_count'] = len(stat['recent_errors'])
    stat['recent_errors'] = (stat['recent_errors'][-10:]) if stat['recent_errors'] else []
    stat['scene_stats'] = {k: _finalize_stat_dict(v) for k, v in stat['scene_stats'].items()}
    return stat


def merge_api_stats_dicts(stats_list: List[Dict]) -> Dict:
    """合并多个 stats dict"""
    merged = _empty_stat_dict()
    for stats in stats_list:
        _merge_stat_dict(merged, stats)
    return _finalize_stat_dict(merged)


def merge_api_stats_files(output_dir: str, pattern: str = "api_stats_training*.json",
                          merged_filename: str = "api_stats_merged.json") -> Tuple[str, Dict]:
    """合并目录下的统计文件并保存"""
    files = sorted(glob.glob(os.path.join(output_dir, pattern)))
    if not files:
        return "", {}

    stats_list = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                stats_list.append(json.load(fp))
        except Exception as e:
            print(f"读取 {f} 失败: {e}")

    if not stats_list:
        return "", {}

    merged = merge_api_stats_dicts(stats_list)
    merged_path = os.path.join(output_dir, merged_filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(merged_path, 'w', encoding='utf-8') as fp:
        json.dump(merged, fp, ensure_ascii=False, indent=2)

    return merged_path, merged


def print_api_stats_from_dict(stats: Dict):
    """使用 APIStats 的打印格式输出合并后的统计"""
    if not stats:
        print("没有可用的 API 统计数据")
        return
    APIStats.from_dict(stats).print_summary()


if __name__ == "__main__":
    # 测试
    setup_api_stats_for_training("test_stats")
    print("测试中... 请手动终止程序查看统计保存")
