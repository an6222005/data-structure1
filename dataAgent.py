import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
import io

# 根據你的專案結構調整下列 import
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

load_dotenv()

# 確保 todo_list.csv 存在，若不存在則建立範例檔案
def initialize_todo_list():
    csv_file_path = "todo_list.csv"
    if not os.path.exists(csv_file_path):
        df = pd.DataFrame(columns=["任務名稱", "截止時間", "重要性", "狀態"])
        df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")
        print(f"已建立 {csv_file_path}，請新增待辦事項。")
# HW1
async def process_chunk(chunk, start_idx, total_records, model_client, termination_condition):
    """
    處理單一批次資料：
      - 轉換為 dict 格式
      - 組出提示詞，請 AI 協助管理待辦事項，
        包括優先順序建議、時間安排建議等。
      - 利用 MultimodalWebSurfer 搜尋時間管理技巧，
        並整合到建議中。
      - 收集所有回覆訊息並返回。
    """
    chunk_data = chunk.to_dict(orient='records')
    prompt = (
        f"目前正在處理第 {start_idx} 至 {start_idx + len(chunk) - 1} 筆待辦事項（共 {total_records} 筆）。\n"
        f"以下為該批次資料:\n{chunk_data}\n\n"
        "請根據以上待辦事項，進行以下分析與建議：\n"
        "  1. 排定優先順序，確保重要且緊急的任務被優先完成；\n"
        "  2. 利用 MultimodalWebSurfer 搜尋最新的時間管理與工作效率提升技巧，\n"
        "     並將搜尋結果整合進回覆中；\n"
        "  3. 提供最佳執行策略，幫助使用者更高效完成待辦事項。\n"
        "請各代理人協同合作，提供完整且有幫助的建議。"
    )
    
    local_data_agent = AssistantAgent("data_agent", model_client)
    local_web_surfer = MultimodalWebSurfer("web_surfer", model_client)
    local_assistant = AssistantAgent("assistant", model_client)
    local_user_proxy = UserProxyAgent("user_proxy")
    local_team = RoundRobinGroupChat(
        [local_data_agent, local_web_surfer, local_assistant, local_user_proxy],
        termination_condition=termination_condition
    )
    
    messages = []
    async for event in local_team.run_stream(task=prompt):
        if isinstance(event, TextMessage):
            print(f"[{event.source}] => {event.content}\n")
            messages.append({
                "batch_start": start_idx,
                "batch_end": start_idx + len(chunk) - 1,
                "source": event.source,
                "content": event.content,
                "type": event.type,
                "prompt_tokens": event.models_usage.prompt_tokens if event.models_usage else None,
                "completion_tokens": event.models_usage.completion_tokens if event.models_usage else None
            })
    return messages

async def main():
    initialize_todo_list()
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("請檢查 .env 檔案中的 GEMINI_API_KEY。")
        return

    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=gemini_api_key,
    )
    
    termination_condition = TextMentionTermination("exit")
    # HW1
    csv_file_path = "todo_list.csv"
    chunk_size = 1000
    chunks = list(pd.read_csv(csv_file_path, chunksize=chunk_size))
    total_records = sum(chunk.shape[0] for chunk in chunks)
    
    tasks = list(map(
        lambda idx_chunk: process_chunk(
            idx_chunk[1],
            idx_chunk[0] * chunk_size,
            total_records,
            model_client,
            termination_condition
        ),
        enumerate(chunks)
    ))
    
    results = await asyncio.gather(*tasks)
    all_messages = [msg for batch in results for msg in batch]
    
    df_log = pd.DataFrame(all_messages)
    output_file = "all_conversation_log.csv"
    df_log.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已將所有對話紀錄輸出為 {output_file}")

if __name__ == '__main__':
    asyncio.run(main())
