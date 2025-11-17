#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys

import pandas as pd


def merge_excels(info_path: Path, perf_path: Path, output_path: Path) -> None:
    """
    将基础信息表与绩效表按“员工ID”做左连接合并，并导出为新的 Excel。

    参数:
        info_path: 基础信息表文件路径，需包含列“员工ID”，以及其它可选字段（姓名/部门等）
        perf_path: 绩效表文件路径，需包含列“员工ID”，以及绩效相关字段（年度/季度/绩效评分等）
        output_path: 合并后导出的 Excel 文件路径
    抛出:
        FileNotFoundError: 任一输入文件不存在
        ValueError: 任一输入表缺少必需列
    """
    if not info_path.exists():
        raise FileNotFoundError(f"基础信息表不存在: {info_path}")
    if not perf_path.exists():
        raise FileNotFoundError(f"绩效表不存在: {perf_path}")

    # 读取 Excel 为 DataFrame
    info_df = pd.read_excel(info_path)
    perf_df = pd.read_excel(perf_path)

    # 基本字段校验：两表都必须包含“员工ID”
    required_info_cols = {"员工ID"}
    required_perf_cols = {"员工ID"}
    missing_info = required_info_cols - set(info_df.columns)
    missing_perf = required_perf_cols - set(perf_df.columns)
    if missing_info:
        raise ValueError(f"基础信息表缺少列: {missing_info}")
    if missing_perf:
        raise ValueError(f"绩效表缺少列: {missing_perf}")

    # 关键逻辑：规范员工ID的数据类型为字符串
    # 目的：避免字符串/数值混用导致 merge 失配（如 1001 vs "1001"）
    info_df["员工ID"] = info_df["员工ID"].astype(str).str.strip()
    perf_df["员工ID"] = perf_df["员工ID"].astype(str).str.strip()

    # 以绩效表为主表做左连接；无法匹配的信息字段填充为 NaN
    merged = perf_df.merge(info_df, on="员工ID", how="left")

    # 关键逻辑：尽量输出更友好的列顺序（存在则前置，不存在则忽略）
    preferred_order = [
        "员工ID",
        "姓名",
        "性别",
        "部门",
        "入职日期",
        "年度",
        "季度",
        "绩效评分",
    ]
    columns = [c for c in preferred_order if c in merged.columns] + [
        c for c in merged.columns if c not in preferred_order
    ]
    merged = merged[columns]

    # 写出到 Excel（使用 openpyxl 引擎以获得更好兼容性）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="Sheet1")

    print(f"合并完成，已生成: {output_path}")


def main(argv=None):
    """
    命令行入口：
      将“员工基本信息表.xlsx”和“员工绩效表.xlsx”按“员工ID”合并为“员工绩效表_1.xlsx”。
    支持通过 --info/--perf/--out 参数覆盖默认路径。
    """
    parser = argparse.ArgumentParser(
        description="将“员工基本信息表.xlsx”和“员工绩效表.xlsx”按员工ID关联并合并为“员工绩效表.xlsx”。"
    )
    parser.add_argument(
        "--info",
        type=Path,
        default=Path("demo_1_merge") / "员工基本信息表.xlsx",
        help="基础信息表路径（默认：demo_1_merge/员工基本信息表.xlsx）",
    )
    parser.add_argument(
        "--perf",
        type=Path,
        default=Path("demo_1_merge") / "员工绩效表.xlsx",
        help="绩效表路径（默认：demo_1_merge/员工绩效表.xlsx）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demo_1_merge") / "员工绩效表_1.xlsx",
        help="输出文件路径（默认：demo_1_merge/员工绩效表_1.xlsx）",
    )
    args = parser.parse_args(argv)

    try:
        merge_excels(args.info, args.perf, args.out)
    except Exception as e:
        print(f"合并失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()




