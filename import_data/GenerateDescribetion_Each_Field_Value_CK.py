#利用大模型，根据表中的几个实例，得到每个字段的取值的描述
from clickhouse_connect import get_client
import json
import re
from openai import OpenAI
from datetime import datetime

from config import Config

# INDEX_NAME = "oh_report_anal_daily_ex"
DOC_LIMIT = 10  # 抓取文档数量
# =====================
# 创建 CK 客户端
# =====================
client = get_client(
    host=Config.CK_HOST,
    port=Config.CK_PORT,
    username=Config.CK_USERNAME,
    password=Config.CK_PASSWORD,
    database=Config.CK_DATABASE
)
# =====================
# 创建 OpenAI 客户端
# =====================
# LLM 客户端
openai_client = OpenAI(
    base_url=Config.VLLM_HOST,
    api_key="EMPTY",
)
llm_model = Config.LLM_MODEL

def convert_dict_to_text(obj):
    if isinstance(obj, dict):
        return {key: convert_dict_to_text(value) for key, value in obj.items()}
    elif isinstance(obj, datetime):
        return obj.isoformat()  # 使用 ISO 8601 格式
    else:
        return str(obj)  # 将其他类型转换为字符串

# =====================
# 通用工具函数
# =====================
def char_list_to_bracket_string(char_list):
    return ",".join([f"[{ch}]" for ch in char_list])

def generate_table_field_comment_json(table_name, field_comment_map):
    """
    生成带表名的 JSON 结构
    """
    if not isinstance(field_comment_map, dict):
        raise ValueError("❌ 输入字段必须为 dict 格式，如 {'字段名': '注释'}")
    result = {
        "table_name": table_name,
        "fields": field_comment_map
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

# =====================
# 从 ClickHouse 抓取样本文档
# =====================
def fetch_documents(table_name, size=10):
    try:
        query = f"SELECT * FROM {table_name} LIMIT {size}"
        result = client.query(query)
        columns = result.column_names
        docs = [dict(zip(columns, row)) for row in result.result_rows]

        # 过滤：仅保留单条长度小于 max_chars 的记录
        max_chars = 1000
        filtered_docs = []
        for doc in docs:
            json_str = convert_dict_to_text(doc)
            total_length = 0
            for key, value in json_str.items():
                if value is not None:  # 排除None值
                    total_length += len(str(value))
                    #print(total_length)

            if total_length <= max_chars:
                filtered_docs.append(doc)

        return filtered_docs

    except Exception as e:
        print(f"抓取表 {table_name} 文档失败: {e}")
        return []

# =====================
# 解析建表语句（简单版）
# =====================
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

# =====================
# 构造大模型 Prompt
# =====================
def generate_prompt(table_name, chinese_name, key_words, documents, english_chinese_field):
    docs_str = json.dumps(documents, ensure_ascii=False, indent=2)
    prompt = f"""
    你是一名精通 ClickHouse 的数据库文档专家。请根据所提供的表结构信息和示例数据，生成一个 JSON 对象，用于描述该表每个字段的取值特性。

    表英文名：{table_name}
    表中文名：{chinese_name}
    核心关键字：{key_words}
    字段名称与中文注释：{english_chinese_field}
    表示例数据（每条为 JSON 对象）：
    {docs_str}

    输出要求：

    1. 输出为 JSON 对象，结构如下：
    {{
      "table_name": "{table_name}",
      "chinese_name": "{chinese_name}",
      "fields": {{
        "字段英文名1": "描述字段取值类型（文字、数字、关键字、时间等）、一般长度、典型格式及特性",
        "字段英文名2": "同上",
        ...
      }}
    }}

    2. 每个字段对应一段自然语言中文说明，专注描述字段的取值类型、一般长度、格式或典型取值，字数不少于 30 字。
    3. 基于示例数据推断字段的取值类型、长度范围和格式特征，但禁止直接列出示例数据。
    4. 输出内容仅为结构化 JSON，严禁 SQL、原始 JSON 数据或字段列表。
    5. 文本描述应简洁、专业、易懂，可直接用于数据字典或技术文档。
    6. 说清楚是否为纯数字或字母和数字的组合。
    请直接输出符合要求的 JSON 文档。
    """
    return prompt

# =====================
# 调用 Ollama
# =====================
# def call_ollama(prompt, model="qwen3-30b-instruct"):
#     try:
#         client = ollama.Client()
#         response = client.generate(model=model, prompt=prompt)
#         return response['response']
#     except Exception as e:
#         print(f"调用 Ollama 失败: {e}")
#         return None

def call_openAI(prompt, model="qwen30ba3", enable_thinking=False):
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            temperature=0,
            top_p=1,
        )
        response = response.choices[0].message.content.strip()
        return response
    except Exception as e:
        print(f"调用 OpenAI 失败: {e}")
        return None

# =====================
# 主程序入口
# =====================
if __name__ == "__main__":

    TableDescribetionDict = {}

    # 这里假设存在一个 metadata 表（或 JSON 文件）保存每个业务表的元信息
    # 例如：table_metadata(英文表名, 中文表名, 使用场景, 表关键字, 建表语句)
    meta_query = f"""SELECT table_name_en, table_name_cn, business_module_lvl1, business_module_lvl2, create_sql, related_tables, table_keywords FROM "default"."table_meta" """
    meta_rows = client.query(meta_query).result_rows

    file_path = "Table_EachField_Descrition_Value.txt"
    All_Text = []
    #test_count = 0
    for row in meta_rows:
        # test_count = test_count + 1
        # if test_count > 5:
        #      break
        INDEX_NAME, INDEX_NAME_CHINESE, BUSSINESS_MODULE_LV1, BUSSINESS_MODULE_LV2, CREATE_SQL, RELATED_TABLES, INDEX_KEY_WORD = row

        print(f"处理表：{INDEX_NAME} ...")
        # 解析建表语句
        fields_info = parse_create_table(CREATE_SQL)

        Chinese_English_dict = {col["name"]: col["comment"] for col in fields_info["columns"]}
        Chinese_Field_String = char_list_to_bracket_string(list(Chinese_English_dict.values()))
        Chinese_English_Sentence = generate_table_field_comment_json(INDEX_NAME, Chinese_English_dict)

        # 抓样本文档
        docs = fetch_documents(INDEX_NAME, DOC_LIMIT)
        if not docs:
            print(f"⚠️ 表 {INDEX_NAME} 未抓取到样本数据，跳过")
            continue

        #jsonDoc = []
        #for eachdoc in docs:
        #    jsonDoc.append(json.dumps(eachdoc))

        jsonDoc = []
        for eachdoc in docs:
            converted_data = convert_dict_to_text(eachdoc)
            jsonDoc.append(json.dumps(converted_data))

        # 构造 Prompt
        prompt = generate_prompt(
            INDEX_NAME,
            INDEX_NAME_CHINESE,
            INDEX_KEY_WORD,
            jsonDoc,
            Chinese_English_Sentence
        )

        # 调用模型
        #result = call_ollama(prompt, model="qwen3:latest")
        byte_length = len(prompt.encode("utf-8"))
        print("UTF-8 字节长度:", byte_length)

        # 计算
        result = call_openAI(prompt,model=llm_model)
        print(result)

        TableDescribetionDict[INDEX_NAME] = result

        if not result:
            continue

        clean_text = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        All_Text.append(clean_text)
        print(f"✅ 已生成：{INDEX_NAME}")

    # 保存输出
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(TableDescribetionDict, f, ensure_ascii=False, indent=2)

    print("🎉 所有表的描述生成完毕！输出路径：", file_path)
