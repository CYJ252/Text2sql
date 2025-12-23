import pandas as pd
from clickhouse_connect import get_client

import sys
import os

# 获取当前脚本所在目录的父目录（即项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config

# 1️⃣ Excel → ClickHouse 列名映射
# -----------------------------
COLUMN_MAP = {
    '序号': 'id',
    '英文表名': 'table_name_en',
    '中文表名': 'table_name_cn',
    '一级业务模块': 'business_module_lvl1',
    '二级业务模块': 'business_module_lvl2',
    '建表语句': 'create_sql',
    '关联表名(多张表,分隔)': 'related_tables',
    '表关键字': 'table_keywords'
}

# -----------------------------
# 3️⃣ 批量插入数据
# -----------------------------
def insert_excel_to_clickhouse(client, table_name: str, df: pd.DataFrame, batch_size=1000):
    """
    将 Excel 读取的 DataFrame 分批插入 ClickHouse 表
    """
    total = len(df)
    rows = []

    for i, row in df.iterrows():
        try:
            seq = int(row.get("序号")) if pd.notna(row.get("序号")) else None
        except (ValueError, TypeError):
            seq = None

        record = (
            seq,
            row.get("英文表名"),
            row.get("中文表名"),
            row.get("一级业务模块"),
            row.get("二级业务模块"),
            row.get("建表语句"),
            row.get("关联表名(多张表,分隔)"),
            row.get("表关键字"),
        )
        rows.append(record)

        # 达到批量上限或最后一行，执行一次批量插入
        if len(rows) >= batch_size or i == total - 1:
            values_sql_parts = []
            for r in rows:
                formatted_values = []
                for v in r:
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        formatted_values.append("NULL")
                    else:
                        # 清理单引号，防止 SQL 错误
                        clean_v = str(v).replace("'", "")
                        formatted_values.append(f"'{clean_v}'")
                values_sql_parts.append("(" + ", ".join(formatted_values) + ")")

            values_sql = ", ".join(values_sql_parts)
            insert_sql = f"""
            INSERT INTO {table_name} (
                id, table_name_en, table_name_cn, 
                business_module_lvl1, business_module_lvl2,
                create_sql, related_tables, table_keywords
            ) VALUES {values_sql}
            """

            client.command(insert_sql)
            rows.clear()

    print(f"✅ 成功插入 {total} 条记录到 {table_name}")
# -----------------------------
# 4️⃣ 主函数
# -----------------------------
def create_and_insert_from_excel(client, file_path: str, table_name: str, sheet_name=0):
    create_sql = """CREATE TABLE IF NOT EXISTS table_meta (
    id Int32 COMMENT '序号',
    table_name_en String COMMENT '英文表名',
    table_name_cn String COMMENT '中文表名',
    business_module_lvl1 String COMMENT '一级业务模块',
    business_module_lvl2 String COMMENT '二级业务模块',
    create_sql String COMMENT '建表语句',
    related_tables String COMMENT '关联表名，多张表用逗号分隔',
    table_keywords String COMMENT '表关键字'
) ENGINE = MergeTree()
ORDER BY id;"""
    print("创建表 SQL:\n", create_sql)
    client.command(create_sql)
    print(f"表 {table_name} 创建成功！")

# -----------------------------
# 5️⃣ 使用示例
# -----------------------------
if __name__ == "__main__":

    client = get_client(
        host=Config.CK_HOST,
        port=Config.CK_PORT,
        username=Config.CK_USERNAME,
        password=Config.CK_PASSWORD,
        database=Config.CK_DATABASE
    )

    df = pd.read_excel(os.getcwd() + "/import_data/数据库表整理汇总.xlsx",
                       sheet_name="CK测试表（电子履历+大修分析）")

    create_and_insert_from_excel(client, os.getcwd() + '/import_data/数据库表整理汇总.xlsx', 'table_meta', "CK测试表（电子履历+大修分析）")

    insert_excel_to_clickhouse(client, table_name = 'table_meta', df = df)

