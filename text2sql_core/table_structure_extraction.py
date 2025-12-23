import pandas as pd

def table_info_to_llm_context(csv_file, table_names_str):
    """
    将表信息转换为适合LLM的上下文文本
    """
    # 常见的中文编码
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'cp936']
    
    for encoding in encodings:
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file, encoding=encoding)
            
            # 处理输入的表名字符串
            table_names = [name.strip() for name in table_names_str.split(',')]
            
            # 假设表名列名为'表名'，如果没有则使用第一列
            if '英文表名' in df.columns:
                table_column = '英文表名'
            else:
                table_column = df.columns[0]
            
            # 过滤数据
            result_df = df[df[table_column].isin(table_names)]
            
            if result_df.empty:
                return "未找到匹配的表信息"
            
            # 转换为文本
            text_parts = [f"相关表信息:\n"]
            for _, row in result_df.iterrows():
                table_name = row[table_column]
                text_parts.append(f"英文表名: **{table_name}**")
                
                for col in df.columns:
                    if col != table_column:
                        value = row[col]
                        if pd.isna(value):
                            value = "空"
                        text_parts.append(f"{col}: {value}")
                
                text_parts.append("")  # 空行
            
            return "\n".join(text_parts)
            
        except Exception:
            continue
    
    return "无法读取文件，请检查文件路径和编码"


if __name__ == "__main__":
    # 使用示例
    csv_file_path = "my_knowledge_base/sql/入湖总表_.csv"
    table_names_input = "oh_report_anal_aft_monitor_detail,oh_tmpl_info,oh_report_info"

    result_text = table_info_to_llm_context(csv_file_path, table_names_input)
    print(result_text)

    # # 保存到文件
    # with open('表信息_LLM上下文.txt', 'w', encoding='utf-8') as f:
    #     f.write(result_text)
    # print("结果已保存到 '表信息_LLM上下文.txt'")