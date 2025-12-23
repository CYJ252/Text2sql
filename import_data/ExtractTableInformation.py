import os
import re
import sys
import pandas as pd
import clickhouse_connect

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from config import Config

# python 文件功能：
# 把表中的所有字段与表1和1对应，且和表名对应起来，用于
# 就是要一一对应，然后关键字和表名对应起来，所有的关键字都是表的注释和表的字段的注释
# 表字段的注释的话。就是根据问用户问题的关键字，直接在clean house里面做这种SQL的查询

DataXLSX = os.getcwd() + "/import_data/数据库表整理汇总.xlsx"
# 提取表名，字段名，Comment，存入 ClickHouse 用于检索

# 连接 ClickHouse 服务器
ck_client = clickhouse_connect.get_client(
    host=Config.CK_HOST,   # 数据库主机地址
    port=Config.CK_PORT,            # HTTP 接口端口（默认8123）
    username=Config.CK_USERNAME,     # 用户名
    password=Config.CK_PASSWORD,    # 密码（如果有）
    database=Config.CK_DATABASE    # 默认数据库
)

# ========== 1. 获取表结构信息 ==========
def parse_create_table(sql: str):
    """
    解析 SQL 建表语句，提取表名、字段、主键、索引、字段类型。
    """
    result = {
        "table_name": None,
        "columns": [],
        "primary_keys": [],
        "key":[]
    }

    # 1. 提取表名
    table_match = re.search(r"CREATE\s+TABLE\s+`?(\w+)`?", sql, re.IGNORECASE)
    if table_match:
        result["table_name"] = table_match.group(1)

    # 提取表注释
    table_comment_match = re.search(r"COMMENT\s+'([^']*)'", sql, re.IGNORECASE)
    if table_comment_match:
        result["table_comment"]  = table_comment_match.group(1)


    # 2. 提取字段定义（括号内内容）
    fields_section = re.search(r"\((.*)\)", sql, re.S)
    if not fields_section:
        return result
    fields_text = fields_section.group(1)

    # 按行分割字段
    lines = [line.strip().strip(",") for line in fields_text.split("\n") if line.strip()]

    for line in lines:
        # 跳过表级约束（PRIMARY KEY 等）
        if line.upper().startswith("PRIMARY KEY"):
            pk_match = re.findall(r"`(\w+)`", line)
            result["primary_keys"].extend(pk_match)
            continue
        if line.upper().startswith("KEY"):
            continue  # 跳过索引定义

        # 提取字段定义（兼容带引号和不带引号的COMMENT）
        field_match = re.match(
            r"`(?P<name>\w+)`\s+(?P<type>[^'\s]+)(.*?COMMENT\s+['\"]?(?P<comment>[^'\"]*)['\"]?)?",
            line,
            re.IGNORECASE
        )
        if field_match:
            col = {
                "name": field_match.group("name"),
                "type": field_match.group("type"),
                "comment": field_match.group("comment") if field_match.group("comment") else "",
                "not_null": "NOT NULL" in line.upper()
            }
            result["columns"].append(col)

            # 如果是行内主键（例如 `id` INT PRIMARY KEY）
            if "PRIMARY KEY" in line.upper():
                result["primary_keys"].append(col["name"])

    return result

def Add_Key(dict_name, key , value):
    if key in dict_name:
        dict_name[key].append(value)
    else:
        dict_name[key] = [value]
    return 1

df2 = pd.read_excel(DataXLSX, sheet_name='CK测试表（电子履历+大修分析）')

#Extract SQL language
SQL_Language = df2['建表语句'].tolist()
Related_Tables = df2['关联表名(多张表,分隔)'].tolist()
# KeyWordTables = df2['表关键字'].tolist()

All_dict = {}

for count in range(len(SQL_Language)):
    each_SQL_Language = SQL_Language[count]
    SQL_Result = parse_create_table(each_SQL_Language)
    Table_Comment = SQL_Result["table_comment"]
    Columns = SQL_Result["columns"]
    Table_Name = SQL_Result["table_name"]

    #做成1和1对应，根据关键字就可以找到表
    # 如果键存在，就追加；否则新建列表
    # 加入关键字： Table Comment - > Table_Name
    # 加入关键字： Field Comment - > Table_Name
    Add_Key(All_dict, Table_Comment, Table_Name)

    for each_columns in Columns:
        Add_Key(All_dict, each_columns["comment"], Table_Name)

    # 存入 ClickHouse:

print(All_dict)

# ========== 如果表存在则删除 ==========
drop_table_sql = "DROP TABLE IF EXISTS keyword_table_mapping;"
ck_client.command(drop_table_sql)
print("🗑️ 已删除旧表：keyword_table_mapping")

# ========== 创建新表 ==========
create_table_sql = """
CREATE TABLE keyword_table_mapping (
    keyword String,
    tables Array(String)
)
ENGINE = MergeTree()
ORDER BY keyword;
"""
ck_client.command(create_table_sql)
print("✅ 已重新创建表：keyword_table_mapping")

# ========== 插入新数据 ==========
data_to_insert = [(key, value_list) for key, value_list in All_dict.items()]
ck_client.insert('keyword_table_mapping', data_to_insert, column_names=['keyword', 'tables'])
print("✅ 数据已成功写入 ClickHouse")
