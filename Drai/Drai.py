import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

# 載入 .env 中的 GEMINI_API_KEY
load_dotenv()

# 定義日記的情感分析項目
ITEMS = [
    "情感正向",
    "情感負向",
    "情感中立",
    "積極行為",
    "消極行為",
    "反思",
    "目標設置"
]

def parse_response(response_text):
    """
    嘗試解析 Gemini API 回傳的 JSON 格式結果。
    """
    cleaned = response_text.strip()
    
    try:
        result = json.loads(cleaned)
        for item in ITEMS:
            if item not in result:
                result[item] = ""
        return result
    except Exception as e:
        print(f"解析 JSON 失敗：{e}")
        print("原始回傳內容：", response_text)
        return {item: "" for item in ITEMS}

def process_diary_entry(client, diary_entries: list, delimiter="-----"):
    """
    將多筆日記內容合併成一個批次請求。
    提示中要求模型對每筆日記內容產生情感分析回應。
    """
    prompt = (
        "你是一位情感分析專家，請根據以下編碼規則評估每篇日記的情感，\n"
        + "\n".join(ITEMS) +
        "\n\n請依據評估結果，對每個項目：若觸及則標記為 1，否則留空。"
        " 請對每篇日記產生 JSON 格式回覆，並在各筆結果間用下列分隔線隔開：\n"
        f"{delimiter}\n"
        "例如：\n"
        "```json\n"
        "{\n  \"情感正向\": \"1\",\n  \"情感負向\": \"\",\n  ...\n}\n"
        f"{delimiter}\n"
        "{{...}}\n```"
    )
    batch_text = f"\n{delimiter}\n".join(diary_entries)
    content = prompt + "\n\n" + batch_text

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content
        )
    except ServerError as e:
        print(f"API 呼叫失敗：{e}")
        return [{item: "" for item in ITEMS} for _ in diary_entries]
    
    print("批次 API 回傳內容：", response.text)
    parts = response.text.split(delimiter)
    results = []
    for part in parts:
        part = part.strip()
        if part:
            results.append(parse_response(part))
    return results

def main():
    # 提示使用者輸入日記檔案
    input_file = input("請輸入日記檔案路徑: ")
    output_file = "processed_diary.csv"
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    if not os.path.exists(input_file):
        print("找不到檔案，請確認檔案路徑")
        return
    
    # 讀取日記檔案
    with open(input_file, "r", encoding="utf-8") as f:
        diary_entries = f.readlines()

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("請設定環境變數 GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)
    
    batch_size = 5
    total = len(diary_entries)
    
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = diary_entries[start_idx:end_idx]
        batch_results = process_diary_entry(client, batch)
        
        # 把結果寫入 CSV
        with open(output_file, "a", encoding="utf-8-sig") as f:
            for entry, result in zip(batch, batch_results):
                entry_result = ",".join([result.get(item, "") for item in ITEMS])
                f.write(f"{entry.strip()},{entry_result}\n")
        
        print(f"已處理 {end_idx} 筆 / {total}")
        time.sleep(1)
    
    print("全部處理完成。最終結果已寫入：", output_file)

if __name__ == "__main__":
    main()
