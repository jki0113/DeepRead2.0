import os
import aiohttp
import asyncio
from log.logger_config import logger, log_execution_time, log

language_code = {
    "Google": {
        "Korean": "ko",
        "English": "en",
        "Chinese": "zh-CN",
        # "Chinese (Simplified)": "zh-CN",
        # "Chinese (Traditional)": "zh-TW",
        "Japanese": "ja",
    },
    "DeepL": {
        "Korean": "KO",
        "English": "EN",
        "Chinese": "ZH",
        # "Chinese (Simplified)": "ZH",
        # "Chinese (Traditional)": "ZH-TW",
        "Japanese": "JA",
    }
}

async def process_deepl_translate_from_text_list(text, target_language, session, language_code):
    url = "https://api.deepl.com/v2/translate"
    auth_key = os.getenv('DEEPL_KEY')

    params = {
        "auth_key": auth_key,
        "text": text,
        "target_lang": language_code['DeepL'][target_language]
    }

    async with session.post(url, params=params) as response:
        if response.status == 200:
            data = await response.json()
            return data["translations"][0]["text"]
        else:
            return f"Error: {response.status}"
        
async def translate_deepl(text_list, target_language):
    async with aiohttp.ClientSession() as session:
        tasks = [process_deepl_translate_from_text_list(text, target_language, session, language_code) for text in text_list]
        translated_texts = await asyncio.gather(*tasks)
        return translated_texts