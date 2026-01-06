import pandas as pd
import re
import clickhouse_connect
from langchain_core.documents import Document

# ========== 1. 获取表结构信息 ==========
def load_table_meta_as_docs(ck_client, meta_table="sap.table_meta", source_name="clickhouse:table_meta"):
    """
    读取 ClickHouse 的 table_meta，并转换为 LangChain Document 列表。
    生成的 Document 结构类似你从 CSVLoader 得到的：
      - metadata: {'source': ..., 'row': 行号, ...}
      - page_content: 把各列拼成可检索文本
    """
    sql = f"SELECT * FROM {meta_table}"
    res = ck_client.query(sql)

    col_names = list(res.column_names)
    rows = res.result_rows

    docs = []
    for i, r in enumerate(rows):
        row_dict = {col_names[j]: r[j] for j in range(len(col_names))}

        text = (
            f"英文表名: {row_dict.get('table_name_en', '')}\n"
            f"中文表名: {row_dict.get('table_name_cn', '')}\n"
            f"备注（原始备注，供参考）: {row_dict.get('raw_remark', '')}\n"
            f"使用场景（原始）: {row_dict.get('usage_scenarios', '')}\n"
            f"一级业务模块: {row_dict.get('business_module_lvl1', '')}\n"
            f"二级业务模块: {row_dict.get('business_module_lvl2', '')}\n"
            f"字段映射: {row_dict.get('field_mapping', '')}\n"
            f"业务对象: {row_dict.get('biz_object', '')}\n"
            f"业务粒度: {row_dict.get('biz_granularity', '')}\n"
            f"主业务键: {row_dict.get('primary_key_fields', '')}\n"
            f"主时间字段: {row_dict.get('time_field', '')}\n"
            f"关联表名(多张表用逗号分隔): {row_dict.get('related_tables', '')}\n"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name, 
                    "row": i,
                },
            )
        )
    return docs

if __name__ == "__main__":
    # 使用示例
    csv_file_path = "my_knowledge_base/sql/入湖总表_.csv"
    table_names_input = "oh_report_anal_aft_monitor_detail,oh_tmpl_info,oh_report_info"
    ck_client = clickhouse_connect.get_client(
        host='127.0.0.1',
        port=8123,      
        username='default',
        password='12345678',
        database='sap'
    )
    # table_names = [name.strip() for name in table_names_input.split(",") if name.strip()]
    all_tables_info = load_table_meta_as_docs(ck_client)
    print("提取到的表信息如下：")
