import pandas as pd
from pathlib import Path
import pytest

from merge_xls import merge_excels


def _write_excel(path: Path, df: pd.DataFrame) -> None:
    """测试辅助：将 DataFrame 写入 Excel（覆盖写入）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)


def _read_excel(path: Path) -> pd.DataFrame:
    """测试辅助：读取 Excel 为 DataFrame。"""
    return pd.read_excel(path)


def test_merge_success(tmp_path: Path):
    """验证标准场景：左连接合并 + 列顺序前置 + 缺失信息留空。"""
    # 基础信息表
    info_df = pd.DataFrame(
        {
            "员工ID": ["1001", "1002", "1003"],
            "姓名": ["张三", "李四", "王五"],
            "性别": ["男", "女", "男"],
            "部门": ["技术部", "市场部", "人事部"],
            "入职日期": ["2020-01-01", "2021-03-15", "2019-07-20"],
        }
    )
    # 绩效表
    perf_df = pd.DataFrame(
        {
            "员工ID": ["1001", "1002", "9999"],
            "年度": [2024, 2024, 2024],
            "季度": ["Q1", "Q1", "Q1"],
            "绩效评分": [4.5, 3.8, 2.0],
        }
    )

    info_path = tmp_path / "员工基本信息表.xlsx"
    perf_path = tmp_path / "员工绩效表.xlsx"
    out_path = tmp_path / "员工绩效表_合并.xlsx"

    _write_excel(info_path, info_df)
    _write_excel(perf_path, perf_df)

    merge_excels(info_path, perf_path, out_path)

    assert out_path.exists()
    merged = _read_excel(out_path)

    # 校验行数与顺序（以绩效表为驱动）
    # 注意：Excel 读入可能将 ID 推断为数值，因此统一转为字符串比较
    assert list(merged["员工ID"].astype(str)) == ["1001", "1002", "9999"]
    # 校验左连接带来的空值
    row_9999 = merged[merged["员工ID"].astype(str) == "9999"].iloc[0]
    assert pd.isna(row_9999.get("姓名"))
    assert pd.isna(row_9999.get("部门"))

    # 简单校验优先列是否在前（若存在）
    preferred_prefix = ["员工ID", "姓名", "性别", "部门", "入职日期", "年度", "季度", "绩效评分"]
    actual_cols = list(merged.columns)
    assert actual_cols[: len(preferred_prefix)] == preferred_prefix


def test_id_type_normalization(tmp_path: Path):
    """验证关键逻辑：员工ID类型规范化（数值/字符串混用也能正确关联）。"""
    # info 的 ID 为字符串，perf 的 ID 为数值，最终应能成功匹配
    info_df = pd.DataFrame(
        {
            "员工ID": ["1001", "1002"],
            "姓名": ["张三", "李四"],
        }
    )
    perf_df = pd.DataFrame(
        {
            "员工ID": [1001, 1002],  # 数值类型
            "绩效评分": [5.0, 4.0],
        }
    )

    info_path = tmp_path / "info.xlsx"
    perf_path = tmp_path / "perf.xlsx"
    out_path = tmp_path / "out.xlsx"
    _write_excel(info_path, info_df)
    _write_excel(perf_path, perf_df)

    merge_excels(info_path, perf_path, out_path)
    merged = _read_excel(out_path)
    # 统一转为字符串比较，避免 Excel 推断类型差异
    assert list(merged["员工ID"].astype(str)) == ["1001", "1002"]
    assert list(merged["姓名"]) == ["张三", "李四"]


def test_missing_files(tmp_path: Path):
    """验证异常：输入文件不存在时抛出 FileNotFoundError。"""
    info_path = tmp_path / "not_exists_info.xlsx"
    perf_path = tmp_path / "not_exists_perf.xlsx"
    out_path = tmp_path / "out.xlsx"

    with pytest.raises(FileNotFoundError):
        merge_excels(info_path, perf_path, out_path)


def test_missing_required_columns_in_info(tmp_path: Path):
    """验证异常：基础信息表缺少必需列“员工ID”时抛出 ValueError。"""
    # 缺少员工ID
    info_df = pd.DataFrame({"姓名": ["张三"]})
    perf_df = pd.DataFrame({"员工ID": ["1001"]})

    info_path = tmp_path / "info.xlsx"
    perf_path = tmp_path / "perf.xlsx"
    out_path = tmp_path / "out.xlsx"
    _write_excel(info_path, info_df)
    _write_excel(perf_path, perf_df)

    with pytest.raises(ValueError):
        merge_excels(info_path, perf_path, out_path)


def test_missing_required_columns_in_perf(tmp_path: Path):
    """验证异常：绩效表缺少必需列“员工ID”时抛出 ValueError。"""
    info_df = pd.DataFrame({"员工ID": ["1001"], "姓名": ["张三"]})
    perf_df = pd.DataFrame({"年度": [2024]})

    info_path = tmp_path / "info.xlsx"
    perf_path = tmp_path / "perf.xlsx"
    out_path = tmp_path / "out.xlsx"
    _write_excel(info_path, info_df)
    _write_excel(perf_path, perf_df)

    with pytest.raises(ValueError):
        merge_excels(info_path, perf_path, out_path)



