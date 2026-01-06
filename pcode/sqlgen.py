import os
import re
from datetime import datetime
from openai import OpenAI
import clickhouse_connect
import jieba
import json
import pandas as pd
from config import Config


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

def get_table_info(ck_client, table_name: str, sample_size: int = 5):
    """
    根据表名获取表结构及样例数据。
    """
    info = {}

    # 1. 获取列名
    columns_query = f"DESCRIBE TABLE {table_name}"
    result = ck_client.query(columns_query)
    columns = result.column_names
    rows = result.result_rows
    meta = [dict(zip(columns, row)) for row in rows]
    # print(json.dumps(meta, ensure_ascii=False, indent=4))
    info["columns"] = meta

    # 2. 获取样例数据
    sample_query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
    sample_result = ck_client.query(sample_query)
    sample_columns = sample_result.column_names
    sample_rows = sample_result.result_rows

    # 3. 转成字典列表
    sample_data = []
    for row in sample_rows:
        row_dict = {}
        for col, val in zip(sample_columns, row):
            if isinstance(val, datetime):
                row_dict[col] = val.strftime("%Y-%m-%d %H:%M:%S") # 处理 datetime
            else:
                row_dict[col] = val
        sample_data.append(row_dict)

    # 4. 字典列表变成文字
    lines = []
    for i, row in enumerate(sample_data, 1):
        fields_text = " | ".join(f"{k}: {v}" for k, v in row.items())
        lines.append(f"Row {i}: {fields_text}")
    info["sample"] = lines

    return info

def extract_all_tables_info(ck_client):
    """
    批量提取所有表信息。
    """
    query = "SELECT * FROM sap.table_meta"
    result = ck_client.query(query)
    columns = result.column_names
    rows = result.result_rows

    col_mapping = {
        "id": "序号",
        "table_name_en": "英文表名",
        "table_name_cn": "中文表名",
        "business_module_lvl1": "一级业务模块",
        "business_module_lvl2": "二级业务模块",
        "create_sql": "建表语句",
        "related_tables": "关联表名",
        "table_keywords": "表关键字",
    }
    cn_columns = [col_mapping[c] for c in columns]
    table_info = [dict(zip(cn_columns, row)) for row in rows]
    # print(json.dumps(table_info, ensure_ascii=False, indent=4))

    for meta in table_info:
        # 根据 建表语句 得到
        parsed = parse_create_table(meta.get("建表语句", ""))
        meta["字段映射"] = {col["name"]: col["comment"] for col in parsed["columns"]}
        # 查询具体表数据得到
        info = get_table_info(ck_client, meta["英文表名"], sample_size=3)
        meta["字段类型"] = {col["name"]: col["type"] for col in info["columns"]}
        meta["样例数据"] = info["sample"]
        # print(json.dumps(meta, ensure_ascii=False, indent=4))

    return table_info

def extract_some_tables_info(ck_client, selected_tables, sample_num = 3):
    """
    批量提取所有表信息。
    """
    query = "SELECT * FROM table_meta WHERE table_name_en IN (%s)" % ','.join(
        ["'%s'" % table for table in selected_tables])

    result = ck_client.query(query)
    columns = result.column_names
    rows = result.result_rows

    col_mapping = {
        "id": "序号",
        "table_name_en": "英文表名",
        "table_name_cn": "中文表名",
        "raw_remark": "备注（原始备注，供参考）",
        "usage_scenarios": "使用场景（原始）",
        "business_module_lvl1": "一级业务模块",
        "business_module_lvl2": "二级业务模块",
        "create_sql": "建表语句",
        "related_tables": "关联表名",
        "biz_object": "业务对象",
        "biz_granularity": "业务粒度",
        "primary_key_fields": "主业务键",
        "time_field": "主时间字段",
    }
    needed_columns = list(col_mapping.keys())
    columns_str = ", ".join([f"`{col}`" for col in needed_columns])
    in_clause = ", ".join([f"'{table}'" for table in selected_tables])
    query = f"SELECT {columns_str} FROM table_meta WHERE table_name_en IN ({in_clause})"

    result = ck_client.query(query)
    columns = result.column_names  # 应该等于 needed_columns（顺序可能不同）
    rows = result.result_rows

    # ✅ 使用实际返回的列顺序来映射中文名（更健壮）
    cn_columns = [col_mapping[col] for col in columns]
    table_info = [dict(zip(cn_columns, row)) for row in rows]

    # 附加样例数据
    for meta in table_info:
        info = get_table_info(ck_client, meta["英文表名"], sample_size=sample_num)
        meta["样例数据"] = info["sample"]

    return table_info

# ========== 2. 根据表信息和用户问题构建 prompt ==========

# 基地编码
base_code_map = {
    "DY": "大亚湾",
    "HY": "红沿河",
    "YJ": "阳江",
    "ND": "宁德",
    "FC": "防城港",
    "TS": "台山",
    "LF": "陆丰",
    "CN": "苍南",
    "HZ": "惠州",
}

# 机组编码
unit_code_map = {
    "DY1": "大亚湾1号机", "DY2": "大亚湾2号机",
    "FC1": "防城港1号机", "FC2": "防城港2号机",
    "FC3": "防城港3号机", "FC4": "防城港4号机",
    "HY0": "红沿河0号机", "HY1": "红沿河1号机", "HY2": "红沿河2号机",
    "HY3": "红沿河3号机", "HY4": "红沿河4号机", "HY5": "红沿河5号机", "HY6": "红沿河6号机",
    "LA1": "岭澳1号机", "LA2": "岭澳2号机", "LA3": "岭澳3号机", "LA4": "岭澳4号机",
    "ND1": "宁德1号机", "ND2": "宁德2号机", "ND3": "宁德3号机", "ND4": "宁德4号机",
    "TS0": "台山0号机", "TS1": "台山1号机", "TS2": "台山2号机", "TS9": "台山9号机",
    "YJ1": "阳江1号机", "YJ2": "阳江2号机", "YJ3": "阳江3号机",
    "YJ4": "阳江4号机", "YJ5": "阳江5号机", "YJ6": "阳江6号机",
}

#读取文件
Table_Field_Describtion_Value_File = "/mnt/sda/PythonProject/CYJ_Project/text2sql/my_rag_vllm/11.27/Table_EachField_Descrition_Value.txt"
with open(Table_Field_Describtion_Value_File, "r", encoding="utf-8") as f:
    data_value_describe = json.load(f)

#all_tables_info 表格信息, user_question, 用户问题， sample_num, 样例数量
def build_llm_prompt(all_tables_info, user_question, sample_num=1):
    """
    根据 all_tables_info 构造用于 LLM 生成 SQL 的 Prompt。
    
    参数：
    - all_tables_info: extract_all_tables_info 返回的表信息列表
    - user_question: 用户自然语言问题
    
    返回：
    - prompt: 字符串，直接可给 LLM
    """
    prompt_lines = [
        "你是一个 ClickHouse SQL 生成专家。请根据以下信息生成 SQL 查询语句。\n"
    ]

    # 基地与机组编码对照表
    prompt_lines.append("\n【基地、机组对照表】")
    prompt_lines.append("\n基地编码：")
    for k, v in base_code_map.items():
        prompt_lines.append(f"  {k} -> {v}")
    prompt_lines.append("\n机组编码：")
    for k, v in unit_code_map.items():
        prompt_lines.append(f"  {k} -> {v}")
    prompt_lines.append("\n")

    # 规则
    prompt_lines.append("【规则】")
    prompt_lines.extend([
        "- 如果没有明确指定，默认会返回 report_code。",
        "- 使用 AS 时需要给别名加反引号'`'，如 AS `报告名称`。",
        "- 关于负责人的查询使用 LIKE。",
        "- oh_tmpl_info不需要包含deleted_flag字段。",
        "- 最终输出的SQL语句需要包含在 <sql>, <\sql>里。",
        "- SQL 语句不能包含表中没有的字段",
        "- 不要输出思考过程",
        "- deleted_flag字段代表此条记录是否逻辑删除。",
        "- 没有deleted_flag字段的表格生成SQL的时候不要加deleted_flag",
        "- create_time字段表示此条记录的创建时间，注意区分。",
        "- update_time字段表示此条记录的修改时间，注意区分。",
        "- SQL语句中都要使用英文字段。"
        "- 没有显式在问题中出现的如[大修后跟踪],[大修前分析],[大修中分析]等查询要求，在SQL语句中不要添加progress字段为相应条件，不能作推测。"
        f"- 现在的时间是 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}。",
    ])

    # 表结构信息
    prompt_lines.append("\n【表结构】")
    for table in all_tables_info:
        prompt_lines.append(f"表: {table['英文表名']} ({table['中文表名']})")
        # prompt_lines.append(f"表关键字: {table.get('表关键字','')}")
        prompt_lines.append("字段:")
        for field, comment in table.get("字段映射", {}).items():

            # 加入取值范围：
            ValueCommentTable = data_value_describe[table['英文表名']]
            Table_Value_Data = json.loads(ValueCommentTable)
            Table_Value_Comment = Table_Value_Data["fields"][field]
            prompt_lines.append(f"  - {field} ({comment}) ({Table_Value_Comment})")
            # prompt_lines.append(table['英文表名'])
            # ftype = table["字段类型"].get(field, "unknown")
            # prompt_lines.append(f"  - {field} ({comment}) [类型: {ftype}]")
        # prompt_lines.append("关联表: " + ", ".join(table.get("关联表", [])) or "无")
        prompt_lines.append("样例数据:")
        for ex in table.get("样例数据", [])[:sample_num]:
            prompt_lines.append(f"  {ex}")
        prompt_lines.append("\n")  # 表间空行

    # 用户问题
    prompt_lines.append(f"用户问题: {user_question}")
    prompt_lines.append(f"输出:")

    return "\n".join(prompt_lines)

# ========== 3. 根据用户问题生成 SQL 语句 ==========
def generate_sql_with_llm(client, llm_model, all_tables_info, user_question, enable_thinking=True, sample_num=1):
    """
    使用 LLM 生成 SQL 并执行查询。
    
    参数：
        client: OpenAI 客户端对象
        llm_model: 使用的 LLM 模型名称
        all_tables_info: 表结构信息，用于生成 Prompt
        user_question: 用户自然语言问题字符串
        
    返回：
        sql_query: 生成的 SQL 查询
    """
    # 查询改写
    user_question = re.sub(r'(?<!再)分析中', r"分析中(status:0)", user_question)
    user_question = user_question.replace("结束大修", "结束大修(oh_end_date IS NOT NULL)")
    
    # 1. 构造 Prompt
    prompt = build_llm_prompt(all_tables_info, user_question, sample_num=sample_num)
    #print('---------- Prompt ----------')
    #print(prompt)

    # 2. 调用 LLM 生成 SQL 查询语句
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            temperature=0,
            top_p=1,
        )
        response = response.choices[0].message.content.strip()
        pattern = r'<sql>(.*?)</sql>'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        return matches[0].strip()

    except Exception as e:
        print(f"⚠️ LLM 生成 SQL 时出错: {e}")
        return None, []

# 功能函数, 提取表名
def extract_table_names(sql: str):
    """
    从 SQL 语句中提取表名。
    支持 FROM、JOIN、UPDATE、INTO 等关键字。
    """
    # 将 SQL 转为小写，便于正则匹配
    sql_lower = sql.lower()

    # 匹配 FROM / JOIN / UPDATE / INTO 后的表名（忽略子查询括号）
    pattern = r'\b(?:from|join|update|into)\s+([a-zA-Z0-9_.]+)'

    # 提取匹配结果
    tables = re.findall(pattern, sql_lower)

    # 去重 + 保留原始顺序
    seen = set()
    table_list = []
    for t in tables:
        if t not in seen:
            seen.add(t)
            table_list.append(t)
    return table_list

# ========== 4. 查询 ClickHouse 数据库 ==========
def query_ck(ck_client, openai_client, llm_model, all_tables_info, user_question, enable_thinking=True, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        sql_query = generate_sql_with_llm(
            client=openai_client,
            llm_model=llm_model,
            all_tables_info=all_tables_info,
            user_question=user_question,
            enable_thinking=enable_thinking
        )
        print('---------- SQL ----------')
        print(sql_query)

        try:
            # SQL 1 结果
            result = ck_client.query(sql_query)
            result_df = pd.DataFrame(result.result_rows, columns=result.column_names)
            # 2. 转换为 Markdown 字符串
            # index=False 表示不包含行号，这对于 LLM 阅读更友好
            markdown_table = result_df.to_markdown(index=False)
            return sql_query, markdown_table
            # result2 = ck_client.query(sql_query)
            # rows2 = result2.result_rows

            # return sql_query, rows2

        except Exception as e:
            print(f"⚠️ 查询 SQL 时出错 (第 {retry_count+1} 次): {e}")
            error_message = str(e)

            # 将错误反馈给 LLM 让其修正
            user_question += f"\n上一个 SQL 出错，错误信息如下：{error_message}\n请修正并重新生成 SQL。"

            retry_count += 1
            if retry_count == max_retries:
                print("❌ 多次重试仍失败，停止重试。")
                return sql_query, None

SQL_Keyword_ProtoType= """SELECT *
FROM keyword_table_mapping
WHERE keyword LIKE '%关键字%';"""

#加载自定义字典
# dict_dir = os.getcwd() + "/system_dict.txt"
dict_dir = "/mnt/sda/PythonProject/CYJ_Project/text2sql/my_rag_vllm/11.27/system_dict.txt"

jieba.load_userdict(dict_dir)

#取消低频词汇，语气助词等
jieba.del_word('几份')
jieba.del_word('列出')
jieba.del_word('一下')
jieba.del_word('情况')
jieba.del_word('如何')
jieba.del_word('最近')
jieba.del_word('多少')
jieba.del_word('各有')
jieba.del_word('给出')
jieba.del_word('怎么样')
jieba.del_word('出来')
jieba.del_word('罗列')
jieba.del_word('有没有')
jieba.del_word('看看')
jieba.del_word('不为')
jieba.del_word('时间')
jieba.del_word('处于')
jieba.del_word('输出')
jieba.del_word('哪些')
jieba.del_word('一共')

def ExtractKeyWord_Table(ck_client, question):
    # jieba 分词, 关键字提取
    seg_list = jieba.cut(question, cut_all=False)
    all_keyword = []
    for element in seg_list:
        if len(element) > 1:
            all_keyword.append(element)

    print("question:", question)
    print("all key word" + str(all_keyword))  # 全模式

    # 获得2个或以上的单词
    # 对于每个关键字获得相应的
    All_Tables = []
    for each_keyword in all_keyword:
        SearchSQL = SQL_Keyword_ProtoType.replace("关键字", each_keyword)
        # Get Table
        result = ck_client.query(SearchSQL)
        rows = result.result_rows
        for eachrow in rows:
            Table_Names = eachrow[1]
            All_Tables = All_Tables + Table_Names

    unique_lst = list(set(All_Tables))
    print(unique_lst)
    print("长度：", len(unique_lst))
    return unique_lst




def main():

    openai_client = OpenAI(
        base_url=Config.VLLM_HOST,
        api_key="EMPTY",
    )

    llm_model = Config.LLM_MODEL
    # 连接 ClickHouse 服务器
    ck_client = clickhouse_connect.get_client(
        host='127.0.0.1',   # 数据库主机地址
        # port=8123,            # HTTP 接口端口（默认8123）
        username='default',     # 用户名
        password='12345678',    # 密码（如果有）
        # database='default'    # 默认数据库
    )

    while True:
        user_question = input("\n请输入你的问题：")
        lst = ExtractKeyWord_Table(ck_client,user_question)
        all_tables_info = extract_some_tables_info(ck_client, lst, sample_num=5)
        print("已提取表信息：")
        # sql_query, rows = query_ck(
        #     ck_client=ck_client,
        #     openai_client=openai_client,
        #     llm_model=llm_model,
        #     all_tables_info=all_tables_info,
        #     user_question=user_question,
        #     enable_thinking=False
        # )
        # print('---------- RESULT ----------')
        # print(rows)

if __name__ == "__main__":
    main()
