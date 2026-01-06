import pandas as pd
import clickhouse_connect
from typing import List

# 你提供的英文表名顺序（用于筛选和排序）
TARGET_TABLES = [
    "ads_sap_reform_data_ai",
    "ads_sap_reform_data_related_ai",
    # ...（此处省略，保留你完整的列表）
    "ads_sap_repair_long_text_add"
]

# 替换为你的完整 TARGET_TABLES 列表（建议从上一个脚本复制过来）
# 为节省篇幅，这里用占位符，实际使用时请粘贴完整列表
# （你可以在脚本顶部直接粘贴你之前提供的全部表名）

def read_ordered_excel(excel_path: str, table_col: str = "英文表名") -> pd.DataFrame:
    """读取 Excel 并按 TARGET_TABLES 顺序返回 DataFrame"""
    df = pd.read_excel(excel_path)
    if table_col not in df.columns:
        raise ValueError(f"列 '{table_col}' 不存在。可用列: {list(df.columns)}")
    
    df[table_col] = df[table_col].astype(str).str.strip()
    table_to_order = {name: i for i, name in enumerate(TARGET_TABLES)}
    filtered = df[df[table_col].isin(table_to_order)].copy()
    filtered['sort_key'] = filtered[table_col].map(table_to_order)
    result = filtered.sort_values('sort_key').drop(columns=['sort_key']).reset_index(drop=True)
    return result


def main():
    # === 配置区 ===
    EXCEL_FILE = "/mnt/sda/PythonProject/CYJ_Project/text2sql/my_rag_vllm/rag_12-15/import_data/rag_table_meta.xlsx"      # ← 替换为你的 Excel 路径

    # ClickHouse 连接配置
    CK_HOST = "127.0.0.1"
    CK_PORT = 8123
    CK_USER = "default"
    CK_PASSWORD = "12345678"
    CK_DATABASE = "sap"                     # ← 替换为你的数据库名

    # 目标表名
    TARGET_TABLE = "table_meta"

    # === 步骤 1：读取 Excel 数据 ===
    print("📥 正在读取 Excel 文件...")
    df = pd.read_excel(EXCEL_FILE)


    if df.empty:
        print("❌ 没有匹配到任何表数据，退出。")
        return

    print(f"✅ 成功加载 {len(df)} 行数据。")

    # === 步骤 2：连接 ClickHouse ===
    print("🔌 正在连接 ClickHouse...")
    client = clickhouse_connect.get_client(
        host=CK_HOST,
        port=CK_PORT,
        username=CK_USER,
        password=CK_PASSWORD,
        database=CK_DATABASE
    )

    # === 步骤 3：删除已存在的表（如果存在）===
    print(f"🗑️  检查并删除已存在的表 `{TARGET_TABLE}`（如果存在）...")
    client.command(f"DROP TABLE IF EXISTS {TARGET_TABLE}")

    # === 步骤 4：创建新表 ===
    # 根据 DataFrame 列动态生成表结构（简单映射）
    # 注意：这里假设所有列都是 String 类型（适合元数据）
    # 如果你有日期、整数等，需手动调整类型
    create_query = f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
    id UInt32 COMMENT '序号',
    table_name_en String COMMENT '英文表名',
    table_name_cn String COMMENT '中文表名',
    raw_remark String COMMENT '备注（原始备注，供参考）',
    usage_scenarios String COMMENT '使用场景（原始）',
    business_module_lvl1 String COMMENT '一级业务模块',
    business_module_lvl2 String COMMENT '二级业务模块',
    create_sql String COMMENT '建表语句',
    field_mapping String COMMENT '字段映射',
    biz_object String COMMENT '业务对象',
    biz_granularity String COMMENT '业务粒度',
    primary_key_fields String COMMENT '主业务键',
    time_field String COMMENT '主时间字段',
    related_tables String COMMENT '关联表名，多张表用逗号分隔'
) ENGINE = MergeTree()
ORDER BY id;
    """
    print("🆕 正在创建新表...")
    client.command(create_query)

    # === 步骤 5：插入数据 ===
    print("📤 正在插入数据到 ClickHouse...")
    """
    将 Excel 读取的 DataFrame 分批插入 ClickHouse 表
    """
    batch_size=1000
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
            row.get("备注（原始备注，供参考）"),
            row.get("使用场景（原始）"),
            row.get("一级业务模块"),
            row.get("二级业务模块"),
            row.get("建表语句"),
            row.get("字段映射"),
            row.get("业务对象"),
            row.get("业务粒度"),
            row.get("主业务键"),
            row.get("主时间字段"),
            row.get("关联表名(多张表,分隔)"),
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
            INSERT INTO {TARGET_TABLE} (
                id, table_name_en, table_name_cn, 
                raw_remark, usage_scenarios,
                business_module_lvl1, business_module_lvl2,
                create_sql, field_mapping, biz_object,
                biz_granularity, primary_key_fields, time_field,
                related_tables
            ) VALUES {values_sql}
            """

            client.command(insert_sql)
            rows.clear()

    print(f"✅ 成功插入 {total} 条记录到 {TARGET_TABLE}")


if __name__ == "__main__":
    main()