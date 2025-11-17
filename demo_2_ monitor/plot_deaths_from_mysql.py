#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict

import pymysql
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams


# ----------------------------
# Configuration
# ----------------------------
MYSQL_HOST = '127.0.0.1'
MYSQL_PORT = 3306
MYSQL_USER = 'root'
MYSQL_PASSWORD = '12345678'
MYSQL_DB = 'demo'
MYSQL_TABLE = 'hong_kong_districts_epidemic_20220201_20250322'

OUTPUT_PATH = Path('/Users/wangzhehan/github/cursor_test/demo_2_ monitor/新增死亡_日期_地区_趋势_from_mysql.png')


def configure_chinese_font():
    preferred_fonts = [
        'PingFang SC', 'Heiti SC', 'STHeiti', 'Hiragino Sans GB',
        'Songti SC', 'STSong', 'SimHei', 'Microsoft YaHei',
        'Noto Sans CJK SC', 'Noto Sans CJK', 'WenQuanYi Micro Hei', 'Arial Unicode MS'
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in preferred_fonts:
        if name in installed:
            chosen = name
            break
    if chosen is None:
        for fnt in font_manager.fontManager.ttflist:
            n = fnt.name or ''
            if any(key in n for key in ['PingFang', 'Heiti', 'YaHei', 'Noto', 'Song', 'Hei', 'Hiragino', 'WenQuan']):
                chosen = n
                break
    if chosen:
        rcParams['font.family'] = chosen
    else:
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = preferred_fonts
    rcParams['axes.unicode_minus'] = False
    return chosen or 'fallback_list'


def fetch_deaths_by_date_region():
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DB, autocommit=True
    )
    sql = f"""
        SELECT
            report_date,
            district_name,
            SUM(COALESCE(new_deaths, 0)) AS new_deaths
        FROM `{MYSQL_TABLE}`
        GROUP BY report_date, district_name
        ORDER BY report_date ASC;
    """
    data = []
    with conn.cursor() as cur:
        cur.execute(sql)
        for row in cur.fetchall():
            d, region, deaths = row
            # d 可能是 datetime.date/datetime 或 str
            if isinstance(d, datetime):
                d = d.date()
            elif isinstance(d, str):
                for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d'):
                    try:
                        d = datetime.strptime(d, fmt).date()
                        break
                    except Exception:
                        pass
                if isinstance(d, str):
                    continue
            data.append((d, str(region), float(deaths or 0)))
    conn.close()
    return data


def plot_trend(data, font_used):
    # data: list[(date, region, deaths)]
    all_dates = set()
    all_regions = set()
    series_map = defaultdict(lambda: defaultdict(float))  # region -> date -> value
    for d, r, v in data:
        series_map[r][d] += v
        all_dates.add(d)
        all_regions.add(r)
    dates_sorted = sorted(all_dates)
    regions_sorted = sorted(all_regions)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for region in regions_sorted:
        y = [series_map[region].get(d, 0.0) for d in dates_sorted]
        # 仅绘制非全零的地区
        if any(y):
            ax.plot(dates_sorted, y, label=region, linewidth=1.6)

    ax.set_title('香港各区 新增死亡 按日期/地区趋势（来自MySQL）')
    ax.set_xlabel('日期')
    ax.set_ylabel('新增死亡（人）')
    ax.grid(True, linestyle='--', alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if len(labels) <= 15:
        ax.legend(ncol=3, fontsize=8)
    else:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=7, ncol=1)

    fig.autofmt_xdate()
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    return OUTPUT_PATH


def main():
    font_used = configure_chinese_font()
    data = fetch_deaths_by_date_region()
    out = plot_trend(data, font_used)
    print('FONT::', font_used)
    print('SAVED::', out)


if __name__ == '__main__':
    main()


