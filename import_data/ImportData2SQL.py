import re
import clickhouse_connect
import sqlparse, mmap, os, sys
from time import sleep
import os

client = clickhouse_connect.get_client(
    host='127.0.0.1',
    port=8123,
    username='default',
    password='12345678',
)

sql_path = os.getcwd() + "/大修分析模块-10.13.sql"

def parse_insert_statements3(sql_text):
    """
    提取所有 INSERT INTO 语句
    返回 [(table_name, [values_list]), ...]
    """
    # 匹配 INSERT INTO ... VALUES (...) 语句
    pattern = re.compile(
        r"INSERT INTO\s+`?(\w+)`?\s+VALUES\s*\((.*?)\);",
        re.IGNORECASE | re.DOTALL
    )
    inserts = pattern.findall(sql_text)
    result = []

    for table_name, values_str in inserts:
        values = []
        # 改进版：逐字符扫描，支持 JSON 内部的逗号和转义引号
        buf, in_str, str_char, esc = "", False, "", False
        for ch in values_str:
            if in_str:
                if esc:  # 处理转义符
                    buf += ch
                    esc = False
                elif ch == "\\":
                    buf += ch
                    esc = True
                elif ch == str_char:  # 结束字符串
                    buf += ch
                    in_str = False
                else:
                    buf += ch
            else:
                if ch in ("'", '"'):  # 开始字符串
                    buf += ch
                    in_str = True
                    str_char = ch
                elif ch == ",":
                    values.append(buf.strip())
                    buf = ""
                else:
                    buf += ch
        if buf:
            values.append(buf.strip())

        # 清洗每个值
        clean_values = []
        for val in values:
            if val.upper() == "NULL":
                clean_values.append(None)
            else:
                # 去掉最外层引号
                if (val.startswith("'") and val.endswith("'")) or \
                   (val.startswith('"') and val.endswith('"')):
                    val = val[1:-1]
                clean_values.append(val)
        result.append((table_name, clean_values))
    return result

def generate_clickhouse_insert(data):
    """
    data: list of tuples, 每个 tuple 形如 (table_name, [values...])
    返回: list of INSERT 语句字符串
    """
    insert_statements = []

    for table_name, values in data:
        formatted_values = []
        for v in values:
            if v is None:
                formatted_values.append('NULL')
            elif isinstance(v, (int, float)):
                formatted_values.append(str(v))
            else:
                # 字符串加单引号，并处理单引号转义
                formatted_values.append("'" + str(v).replace("'", "\\'") + "'")
        values_str = ", ".join(formatted_values)
        sql = f"INSERT INTO {table_name} VALUES ({values_str});"
        insert_statements.append(sql)

    return insert_statements

def parse_create_table(sql: str):
    result = {
        "table_name": None,
        "columns": [],
        "primary_keys": []
    }

    # 1. 提取表名
    table_match = re.search(r"CREATE\s+TABLE\s+`?(\w+)`?", sql, re.IGNORECASE)
    if table_match:
        result["table_name"] = table_match.group(1)

    # 提取表注释
    table_comment_match = re.search(r"COMMENT\s*=\s*'([^']+)'", sql, re.IGNORECASE)
    if table_comment_match:
        result["table_comment"] = table_comment_match.group(1)

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

def generate_clickhouse_create_table(table_info: dict) -> str:
    """
    根据表结构字典生成 ClickHouse CREATE TABLE 语句

    Args:
        table_info (dict): 表结构信息，格式如下：
            {
                'table_name': '表名',
                'columns': [
                    {'name': '字段名', 'type': 'varchar(50)', 'comment': '注释', 'not_null': True},
                    ...
                ],
                'primary_keys': ['主键字段名1', '主键字段名2', ...]
            }

    Returns:
        str: ClickHouse CREATE TABLE SQL
    """

    # MySQL 类型 -> ClickHouse 类型映射
    type_map = {
        'varchar': 'String',
        'int': 'Int32',
        'tinyint': 'Int8',
        'datetime': 'DateTime'
    }

    def convert_type(mysql_type: str) -> str:
        """转换 MySQL 类型为 ClickHouse 类型"""
        if mysql_type.startswith('varchar'):
            return 'String'
        return type_map.get(mysql_type, 'String')  # 默认 String

    # 生成字段 SQL
    columns_sql = []
    for col in table_info['columns']:
        col_name = col['name']
        col_type = convert_type(col['type'])
        comment = col.get('comment', '')
        columns_sql.append(f"`{col_name}` {col_type} COMMENT '{comment}'")

    columns_sql_str = ",\n  ".join(columns_sql)

    # 生成主键
    primary_keys = ", ".join([f"`{pk}`" for pk in table_info.get('primary_keys', [])])

    # 拼接最终 SQL
    create_table_sql = f"""
CREATE TABLE {table_info['table_name']} (
  {columns_sql_str}
) ENGINE = MergeTree()
PRIMARY KEY ({primary_keys})
ORDER BY ({primary_keys});
"""
    return create_table_sql



def iter_statements(path):
    """
    逐条 yield 完整 SQL 语句（CREATE/INSERT/...）
    支持 GB 级文件，内存占用 < 10 MB
    """
    buffer = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_strip = line.strip()
            buffer.append(line)
            # 只判断行末分号
            if line_strip.endswith(';'):
                stmt = ''.join(buffer).strip()
                if stmt:
                    yield stmt
                buffer.clear()
        # 文件末尾无分号
        if buffer:
            yield ''.join(buffer).strip()

def parse_one(stmt):
    """单条解析，返回 (kind, data)"""
    parsed = sqlparse.parse(stmt)[0]
    kind = parsed.get_type()
    if kind == 'CREATE':
        try:
            Tables_dict = parse_create_table(stmt)
            clen_sql = generate_clickhouse_create_table(Tables_dict)
            #print(clen_sql)
            #print("执行成功:", clen_sql)
            client.command(clen_sql)
            print("执行成功:", stmt[:50], "...")
        except:
            print(stmt)

    elif kind == 'INSERT':
        a = 1
        #continue

        try:
            StateMents = parse_insert_statements3(stmt)
            clean_sql = generate_clickhouse_insert(StateMents)
            client.command(clean_sql[0])
            #print("执行成功:", stmt[:50], "...")
        except Exception as e:
            print("执行失败:")
            print("原始语句:", stmt)
            if 'StateMents' in locals():
                print("解析结果:", StateMents)
            if 'clean_sql_list' in locals():
                print("生成的 INSERT 语句:", clean_sql)
            print("异常信息:", str(e))
            sleep(20)

        #a = 1
        #client.command(stmt)
        #print("执行成功:", stmt[:50], "...")

    return None, None

for n, stmt in enumerate(iter_statements(sql_path)):
    parse_one(stmt)

print("SQL 文件执行完成")