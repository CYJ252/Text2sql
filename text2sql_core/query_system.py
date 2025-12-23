import asyncio
import json
import logging
import re
from typing import Iterator
from openai import OpenAI
import pandas as pd
import requests
import os
import time
import datetime

# 导入prompt模块
from .prompt import PROMPTS


class QuerySystem:
    def __init__(self, llm_model, vllm_host ,api_key='Empty'):
        self.llm_model = llm_model
        self.vllm_host = vllm_host
        self.vllm_client =OpenAI(
            base_url=self.vllm_host,
            api_key=api_key,
        )

    
    def query_ck(self, user_question, table_info, case_info,max_retries=3,ck_client=None):
        question = user_question
        retry_count = 0
        table_info_str = json.dumps(table_info, ensure_ascii=False, indent=2)
        while retry_count < max_retries:
            sql=self._generate_sql_candidates(question, table_info_str, case_info)

            
            sql = sql.strip()
            pattern = r'```sql(.*?)```'
            matches = re.findall(pattern, sql, re.DOTALL | re.IGNORECASE)
            sql_query =  matches[0].strip()

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

            except Exception as e:
                print(f"⚠️ 查询 SQL 时出错 (第 {retry_count+1} 次): {e}")
                error_message = str(e)

                # 将错误反馈给 LLM 让其修正
                question =user_question+ f"\n上一个 SQL{sql_query} 出错，错误信息如下：{error_message}\n请修正并重新生成 SQL。"

                retry_count += 1
                if retry_count == max_retries:
                    print("❌ 多次重试仍失败，停止重试。")
                    return sql_query, None

    def query(self, question, results, max_retries=3):
        print(f"\n正在处理问题: {question}")

        if not results:
            context_str = "没有找到相关信息。"
        else:
            context_str = results
        
    
        print("步骤2: 组合提示词并调用大语言模型生成答案...")
        for attempt in range(max_retries):
            print(f"\n--- 第 {attempt + 1}/{max_retries} 次尝试 ---")

            # 步骤2: 生成并排序候选SQL
            candidates_text = self._generate_sql_candidates(question, context_str)
            print(candidates_text)
            ranked_sqls = self._parse_and_rank_candidates(question, candidates_text)

            if not ranked_sqls:
                print("未能生成任何候选SQL，尝试继续...")
                continue
            
            print(f"步骤2.3: 获得 {len(ranked_sqls)} 个排序后的候选SQL。")

            # 步骤3: 校验循环
            for i, sql in enumerate(ranked_sqls):
                print(f"\n正在校验排名第 {i+1} 的SQL:")
                print(f"sql\n{sql}\n```")

                    # 3a. 语法校验
                if not self._validate_sql_syntax(sql):
                    continue # 失败，尝试下一个

                # 3b. 语义校验
                if not self._validate_sql_semantics(sql, context_str,question):
                    continue # 失败，尝试下一个
                
                # 如果所有校验都通过
                print("\n🎉 找到一个通过所有校验的有效SQL！")
                final_answer = f"```sql\n{sql}\n```"
                print("\n最终答案:")
                print(final_answer)
                return final_answer # 成功，返回结果并退出函数

            print("\n本次尝试中的所有候选SQL均未通过校验。")

        # 如果所有重试都失败了
        print("\n所有尝试均失败，无法生成有效的SQL。")
        final_answer = "根据提供的资料，我无法生成一个有效的SQL来回答这个问题。请尝试换一种问法或联系技术人员。"
        print(final_answer)
        return final_answer
    
    def html_query(self, question,sql_1, result1,sql_2, result2,SAVE_PATH=None,number=0):
        html_outpot=self.html_explain_result(question,sql_1, result1,sql_2, result2)
        pattern = r'```html(.*?)```'
        matches = re.findall(pattern, html_outpot, re.DOTALL | re.IGNORECASE)
        if SAVE_PATH is not None:
            with open(f'{SAVE_PATH}/{number}.html', 'w', encoding='utf-8') as f:
                f.write(matches[0].strip())
    
    def html_explain_result(self, question,sql_1, result1,sql_2, result2):
        sys_prompt = PROMPTS["Finall_2"]
        def safe_str(value):
            return "" if value is None else str(value)
        sys_prompt = sys_prompt.replace("{question}", safe_str(question))
        sys_prompt = sys_prompt.replace("{sql_1}", safe_str(sql_1))
        sys_prompt = sys_prompt.replace("{res_1_md}", safe_str(result1))
        sys_prompt = sys_prompt.replace("{sql_2}", safe_str(sql_2))
        sys_prompt = sys_prompt.replace("{res_2_md}", safe_str(result2))

        try:
            res = self.vllm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": sys_prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0,
            )
            result=res.choices[0].message.content.strip()
            return result

        except Exception as e:
            print(f"\n\n发生错误: {e}")
            return None

    def generate_json_analysis_2(self, question,sql_1, result1,sql_2, result2):
       # 1. 准备数据：将结果转为 Markdown 格式，利于 LLM 理解
        res_1_md = result1
        res_2_md = result2
        
        # 2. 替换 Prompt 变量
        sys_prompt = PROMPTS["Finall_2"]
        def safe_str(value):
            return "" if value is None else str(value)

        # 使用 replace 而不是 format，避免 JSON 模板中的花括号冲突
        sys_prompt = sys_prompt.replace("{question}", safe_str(question))
        sys_prompt = sys_prompt.replace("{sql_1}", safe_str(sql_1))
        sys_prompt = sys_prompt.replace("{res_1_md}", safe_str(res_1_md))
        sys_prompt = sys_prompt.replace("{sql_2}", safe_str(sql_2))
        sys_prompt = sys_prompt.replace("{res_2_md}", safe_str(res_2_md))
        sys_prompt = sys_prompt.replace("{current_time_str}", datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M"))

        try:
            # 3. 调用 LLM
            res = self.vllm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": sys_prompt}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                temperature=0.1, # 稍微给一点点温度或保持0，JSON格式生成0.1通常比较稳
            )
            llm_output = res.choices[0].message.content.strip()

            # 4. 解析 JSON
            parsed_result = self._parse_llm_json(llm_output)

            if parsed_result:
                # 确保 status 存在
                if "status" not in parsed_result:
                    parsed_result["status"] = "success"
                return parsed_result
            else:
                # 解析失败的兜底返回
                return {
                    "status": "error",
                    "sql": sql_1 or sql_2,
                    "result": None,
                    "message": f"模型返回格式错误，无法解析为JSON。原始内容片段: {llm_output[:50]}..."
                }

        except Exception as e:
            print(f"\n\n发生错误: {e}")
            return {
                "status": "error",
                "sql": sql_1,
                "result": None,
                "message": f"内部处理错误: {str(e)}"
            }

    def generate_json_analysis(self, question,sql, result):
       # 1. 准备数据：将结果转为 Markdown 格式，利于 LLM 理解
        # 2. 替换 Prompt 变量
        sys_prompt = PROMPTS["Finall"]
        def safe_str(value):
            return "None" if (value is None or value == "") else str(value)

        # 使用 replace 而不是 format，避免 JSON 模板中的花括号冲突
        sys_prompt = sys_prompt.replace("{question}", safe_str(question))
        sys_prompt = sys_prompt.replace("{sql}", safe_str(sql))
        sys_prompt = sys_prompt.replace("{query_result}", safe_str(result))
        sys_prompt = sys_prompt.replace("{current_time_str}", datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M"))

        with open('logs/结果分析prompt.txt', 'w', encoding='utf-8') as f:
            f.write(sys_prompt)

        try:
            # 3. 调用 LLM
            res = self.vllm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": sys_prompt}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                temperature=0.1, # 稍微给一点点温度或保持0，JSON格式生成0.1通常比较稳
            )
            llm_output = res.choices[0].message.content.strip()

            # 4. 解析 JSON
            parsed_result = self._parse_llm_json(llm_output)

            if parsed_result:
                # 确保 status 存在
                if "status" not in parsed_result:
                    parsed_result["status"] = "success"
                return parsed_result
            else:
                # 解析失败的兜底返回
                return {
                    "status": "error",
                    "sql": sql,
                    "result": None,
                    "message": f"模型返回格式错误，无法解析为JSON。原始内容片段: {llm_output[:50]}..."
                }

        except Exception as e:
            print(f"\n\n发生错误: {e}")
            return {
                "status": "error",
                "sql": sql,
                "result": None,
                "message": f"内部处理错误: {str(e)}"
            }
        
    def generate_json_analysis_stream(self, question, sql, result) -> Iterator[str]:
        """
        流式生成分析结果
        """
        # 1. 准备数据
        # 使用你修改后的 PROMPTS["Finall_2"]
        sys_prompt = PROMPTS["Finall_stream"] 
        
        def safe_str(value):
            return "" if value is None else str(value)

        sys_prompt = sys_prompt.replace("{question}", safe_str(question))
        sys_prompt = sys_prompt.replace("{sql}", safe_str(sql))
        # 这里的 result 已经是 markdown 表格字符串
        sys_prompt = sys_prompt.replace("{query_result}", safe_str(result))
        sys_prompt = sys_prompt.replace("{current_time_str}", datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M"))

        try:
            # 2. 调用 LLM，开启 stream=True
            stream = self.vllm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": sys_prompt}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                temperature=0.1,
                stream=True  # <--- 关键开启流式
            )

            # 3. 逐步 yield 内容
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            print(f"\n\n流式生成发生错误: {e}")
            # 发生错误时，返回一个符合 JSON 结构的错误信息片段，或者直接抛出
            yield f'{{"status": "error", "message": "Stream error: {str(e)}"}}'

    def _generate_sql_candidates(self, question, context_str, case_str):
        """
        使用CoT生成候选SQL。
        """
        now = datetime.datetime.now()
        current_time_str = now.strftime("%Y年%m月%d日 %H:%M")
        sys_prompt_temp = PROMPTS["generate_SQL"]
        sys_prompt = sys_prompt_temp.format(
            current_time_str=current_time_str,
            question=question,
            context_str=context_str,
            case_str=case_str,
        )

        with open('logs/SQL生成prompt.txt', 'w', encoding='utf-8') as f:
            f.write(sys_prompt)

        res = self.vllm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": sys_prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0,
        )
        result=res.choices[0].message.content.strip()
        return result

    def _parse_and_rank_candidates(self, question, candidates_text):
        """
        解析LLM生成的文本，并对SQL进行排序。
        """
        print("步骤2.2: 解析并对候选SQL进行排序...")
        
        # 使用正则表达式解析出思考和SQL
        pattern = re.compile(r"\[CANDIDATE \d+\]\s*思考: (.*?)\s*SQL:\s*```sql\s*(.*?)\s*```", re.DOTALL)
        matches = pattern.findall(candidates_text)
        
        if not matches:
            print("警告: 无法从LLM的输出中解析出任何候选SQL。")
            # 尝试把整个输出当作一个SQL
            if "```sql" in candidates_text:
                sql_match = re.search(r"```sql\s*(.*?)\s*```", candidates_text, re.DOTALL)
                if sql_match:
                    return [sql_match.group(1).strip()]
            return []

        # 候选SQL列表
        candidate_sqls = [sql.strip() for _, sql in matches]

        # 构建排序Prompt
        prompt_template = f"""
        你是一个SQL评审专家。下面是用户的一个问题和几个由AI生成的候选SQL。按原来得到顺序对它们进行排序。

        [用户问题]
        {question}

        [候选SQL列表]
        """
        for i, sql in enumerate(candidate_sqls):
            prompt_template += f"\n-- SQL {i+1} --\n{sql}\n"

        prompt_template += """
        # 任务:
        请输出一个排序后的索引列表，按原来的顺序进行排序
        只输出数字和逗号，不要有任何其他解释。

        排序索引:
        """
        res = self.vllm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt_template}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0,
        )
        response=res.choices[0].message.content.strip()
        
        try:
            order_str = response['message']['content'].strip()
            # 清理可能的非数字字符
            order_str = re.sub(r'[^\d,]', '', order_str)
            ranked_indices = [int(i.strip()) - 1 for i in order_str.split(',')]
            
            # 根据LLM返回的顺序重新排列SQL
            ranked_sqls = [candidate_sqls[i] for i in ranked_indices if i < len(candidate_sqls)]
            
            # 添加任何未被排序的SQL到末尾，以防LLM排序出错
            for i, sql in enumerate(candidate_sqls):
                if sql not in ranked_sqls:
                    ranked_sqls.append(sql)

            print(f"LLM排序结果: {order_str}")
            return ranked_sqls
        except Exception as e:
            print(f"排序失败: {e}。将使用原始顺序。")
            return candidate_sqls # 如果排序失败，返回原始顺序
        

    def _validate_sql_semantics(self, sql_query, context_str, question):
        """
        使用LLM检查SQL的逻辑。
        """
        sys_prompt_temp = PROMPTS["validate_sql"]
        sys_prompt = sys_prompt_temp.format(
            sql_query=sql_query,
            context_str=context_str,
        )

        res = self.vllm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": sys_prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0,
        )
        answer = res.choices[0].message.content.strip()
        if answer.upper() == "OK":
            print("  ✅ 语义校验通过")
            return True
        else:
            print(f"  ❌ 语义校验失败: {answer}")
            return False 

    # 辅助方法：解析 LLM 返回的 JSON 字符串
    def _parse_llm_json(self, text):
        try:
            # 1. 尝试直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        try:
            # 2. 使用正则提取第一个 { 和最后一个 } 之间的内容（去除 markdown 代码块标记）
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        
        # 3. 解析失败返回 None
        return None
