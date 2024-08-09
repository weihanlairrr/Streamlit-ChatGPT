#%% 導入套件
import streamlit as st
import base64
import pandas as pd
import requests
import asyncio
import json
import time
import re
import os
import html
import streamlit_shadcn_ui as ui

from io import BytesIO
from streamlit_option_menu import option_menu
import pytz
from openai import OpenAI, AsyncOpenAI
from datetime import datetime

#%% 保存和載入設置      
def save_settings(settings):
    with open('settings.json', 'w') as f:
        json.dump(settings, f)

def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_chat_history(chat_history, model_type):
    filename = 'chat_history_gpt.json' if model_type == 'ChatGPT' else ('chat_history_perplexity.json' if model_type == 'Perplexity' else 'chat_history_dalle.json')
    with open(filename, 'w') as f:
        json.dump(chat_history, f)

    placeholder_status = {
        'ChatGPT': st.session_state['gpt_chat_started'],
        'Perplexity': st.session_state['perplexity_chat_started'],
        'DALL-E': st.session_state['dalle_chat_started']
    }
    with open('text_placeholder_status.json', 'w') as f:
        json.dump(placeholder_status, f)

def load_chat_history(model_type):
    filename = 'chat_history_gpt.json' if model_type == 'ChatGPT' else ('chat_history_perplexity.json' if model_type == 'Perplexity' else 'chat_history_dalle.json')
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_placeholder_status():
    try:
        with open('text_placeholder_status.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'ChatGPT': False, 'Perplexity': False, 'DALL-E': False}

def save_shortcuts():
    with open('shortcuts.json', 'w') as f:
        json.dump({
            'shortcuts': st.session_state['shortcuts'],
            'exported_shortcuts': st.session_state.get('exported_shortcuts', [])
        }, f)

def load_shortcuts():
    try:
        with open('shortcuts.json', 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                st.session_state['shortcuts'] = data.get('shortcuts', [])
                st.session_state['exported_shortcuts'] = data.get('exported_shortcuts', [])
            else:
                st.session_state['shortcuts'] = data
                st.session_state['exported_shortcuts'] = []
    except FileNotFoundError:
        st.session_state['shortcuts'] = [{
            "name": "Shortcut 1",
            "components": [],
            "prompt_template": ""
        }]
        st.session_state['exported_shortcuts'] = []

def update_and_save_setting(key, value):
    st.session_state[key] = value
    settings[key] = value
    save_settings(settings)
    
settings = load_settings()
chat_history_gpt = load_chat_history('ChatGPT')
chat_history_perplexity = load_chat_history('Perplexity')
chat_history_dalle = load_chat_history('DALL-E')
load_shortcuts()

#%% 載入圖片
def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

assistant_avatar_gpt = get_image_as_base64("Images/ChatGPT Logo.png")
assistant_avatar_perplexity = get_image_as_base64("Images/Perplexity Logo.png")
user_avatar_default = get_image_as_base64("Images/Cutie.png")
logo_base64 = get_image_as_base64("Images/Bot Logo.png")

avatars = {
    "Cutie": get_image_as_base64("Images/Cutie.png"),
    "Boy": get_image_as_base64("Images/Boy.png"),
    "Penguin": get_image_as_base64("Images/Penguin.png"),
    "Otter": get_image_as_base64("Images/Otter.png"),
    "Bird": get_image_as_base64("Images/Bird.png"),
    "White": get_image_as_base64("Images/White.png"),
    "Bear": get_image_as_base64("Images/Bear.png"),
    "Girl": get_image_as_base64("Images/Girl.png"),
    "Baby Girl": get_image_as_base64("Images/Baby Girl.png"),
    "Dog": get_image_as_base64("Images/Dog.png"),
    "Chinese": get_image_as_base64("Images/Chinese.png"),
    "Monkey": get_image_as_base64("Images/Monkey.png"),
}

#%% 初始化狀態變量
def init_session_state():
    settings_defaults = {
        'chatbot_api_key': settings.get('chatbot_api_key', ''),
        'perplexity_api_key': settings.get('perplexity_api_key', ''),
        'open_ai_model': settings.get('open_ai_model', 'gpt-4o-mini'),
        'perplexity_model': settings.get('perplexity_model', 'llama-3.1-sonar-large-128k-online'),
        'perplexity_temperature': settings.get('perplexity_temperature', 0.5),
        'perplexity_top_p': settings.get('perplexity_top_p', 0.5),
        'perplexity_presence_penalty': settings.get('perplexity_presence_penalty', 0.0),
        'perplexity_max_tokens': settings.get('perplexity_max_tokens', 1000),
        'perplexity_system_prompt': settings.get('perplexity_system_prompt', ''),
        'gpt_system_prompt': settings.get('gpt_system_prompt', ''),
        'language': settings.get('language', '繁體中文'),
        'temperature': settings.get('temperature', 0.5),
        'top_p': settings.get('top_p', 0.5),
        'presence_penalty': settings.get('presence_penalty', 0.0),
        'frequency_penalty': settings.get('frequency_penalty', 0.0),
        'max_tokens': settings.get('max_tokens', 1000),
        'content': '',
        'reset_confirmation': False,
        'gpt_chat_started': False,
        'perplexity_chat_started': False,
        'dalle_chat_started': False,
        'api_key_removed': False,
        'model_type': 'ChatGPT',
        'user_avatar': settings.get('user_avatar', user_avatar_default),
        'prompt_submitted': False,
        'reset_triggered': False,
        'text_placeholder': None,
        'dalle_model': settings.get('dalle_model', 'dall-e-3'),
        'reset_confirmed': False,
        'shortcuts': load_shortcuts(),
        'shortcut':[],
        'current_shortcut': 0,
        'new_component': {"label": "", "options": ""},
        'shortcut_names': [shortcut["name"] for shortcut in st.session_state['shortcuts']],
        'exported_shortcuts': [],
        'avatar_selected': False,
        'expander_state': True,
    }

    for key, default_value in settings_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    if st.session_state.get('reset_triggered', False):
        st.session_state[f'messages_{st.session_state["model_type"]}'] = []
        st.session_state['reset_triggered'] = False
    else:
        if 'messages_ChatGPT' not in st.session_state:
            st.session_state['messages_ChatGPT'] = chat_history_gpt.get('ChatGPT', [])
        if 'messages_Perplexity' not in st.session_state:
            st.session_state['messages_Perplexity'] = chat_history_perplexity.get('Perplexity', [])

    placeholder_status = load_placeholder_status()
    st.session_state['gpt_chat_started'] = placeholder_status.get('ChatGPT', False)
    st.session_state['perplexity_chat_started'] = placeholder_status.get('Perplexity', False)
    st.session_state['dalle_chat_started'] = placeholder_status.get('DALL-E', False)

init_session_state()

#%% 自訂樣式
with st.sidebar:
    st.markdown(
        """
        <style>
        .container {
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-sizing: border-box;
        }
        .chat-container {
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            padding: 0 10px;
            box-sizing: border-box;
        }
        .message-container {
            background: #F1F1F1;
            color: #2B2727;
            border-radius: 15px;
            padding: 10px 15px;
            margin-right: 5px;
            margin-left: 5px;
            font-size: 16px;
            max-width: 100%;
            word-wrap: break-word;
            word-break: break-all;
            box-sizing: border-box;
        }
        .message-container p {
            margin: 1em 0 !important;
        }
        .message-container p:last-child {
            margin-bottom: 0 !important;
        }
        .message-container p:not(:last-child) {
            margin-bottom: 1em !important; 
        }
        .st-chat-input {
            width: 100%;
            box-sizing: border-box;
        }
        .stButton > button {
            padding: 5px 20px;
            background: #DDE3EA;
            color: #2B2727;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
            margin: 5px 0;
            width: 100%;
        }
        .stButton > button:hover {
            background: #C0C0C0;
        }
        .stRadio > div {
            display: flex;
            justify-content: center;
            padding: 0px 0px 0px 18px;
            border: none;
            border-radius: 5px;
            background: linear-gradient(-135deg, #FFFFFF 0%, #ECECEC 80%, #D4D4D4 80%, #ECECEC 80%);
        }
        .message {
            white-space: pre-wrap !important;
        }
        .message-container pre {
            background-color: #1E1E1E !important;
            border-radius: 5px !important;
            padding: 10px !important;
            overflow-x: auto !important;
            margin: 10px 0 !important;
            white-space: pre !important;
        }
        .message-container pre code {
            font-family: 'Source Code Pro', 'Courier New', monospace !important;
            font-size: 16px !important;
            line-height: 1.4 !important;
            white-space: pre !important;
            color: #f1f1f1 !important;
        }
        .message-container code:not(pre code) {
            background: #1E1E1E !important;
            color: #f1f1f1 !important;
            font-size: 13px !important;
            border-radius: 4px !important;
            display: inline-flex !important;
            align-items: center !important;
            padding: 2px 4px !important;
            margin: 2px 2px !important;
        }
        .stCodeBlock button {
            color: white !important;
        }
        .stCodeBlock button svg {
            stroke: white !important;
        }
        .stCodeBlock button:hover {
            color: white !important;
        }
        .stCodeBlock button:hover svg {
            stroke: white !important;
        }
        .fixed-bottom {
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            width: 100% !important;
            background-color: white !important;
            padding: 10px !important;
            border-top: 1px solid #ddd !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            z-index: 9999 !important;
        }
        .text-input {
            flex: 1 !important;
            padding: 10px !important;
            margin-right: 10px !important;
            border-radius: 5px !important;
            border: 1px solid #ddd !important;
        }   
        .btn {
            padding: 10px !important;
            border-radius: 5px !important;
            background-color: #f0f0f0 !important;
            border: 1px solid #ddd !important;
            cursor: pointer !important;
            width: 40px !important;
            height: 40px !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }   
        .message-container p {
            margin: 1em 0 !important;
        }
        .message-container p:last-child {
            margin-bottom: 0 !important;
        }
        .message-container p + ul,
        .message-container p + ol {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        .message-container p + ul + p,
        .message-container p + ol + p {
            margin-top: 1em !important;
        }
        [data-testid=stSidebar] {
            background: #F0F4F8;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
            
#%% 產生對話
def get_openai_response(client, model, messages, temperature, top_p, presence_penalty, frequency_penalty, max_tokens, system_prompt, language):
    try:
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        if st.session_state['language']:
            prompt = messages[-1]['content'] + f" 請使用{st.session_state['language']}回答。"
            messages[-1]['content'] = prompt

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stream=True
        )
        streamed_text = ""
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                streamed_text += chunk_content
                html_chunk = format_message(streamed_text)
                yield html_chunk
    except Exception as e:
        error_message = str(e)
        if "Incorrect API key provided" in error_message:
            yield "請輸入正確的 OpenAI API Key"
        elif "insufficient_quota" in error_message:
            yield "您的 OpenAI API餘額不足，請至您的帳戶加值"
        elif isinstance(e, UnicodeEncodeError):
            yield "請輸入正確的 OpenAI API Key"
        else:
            yield f"Error: {error_message}"

def generate_perplexity_response(prompt, history, model, temperature, top_p, presence_penalty, max_tokens, system_prompt, language):
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {st.session_state['perplexity_api_key']}"
        }

        if st.session_state['language']:
            prompt = prompt + f" 請使用{st.session_state['language']}回答。"

        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        full_prompt = f"{system_prompt}\n\n{context}\nuser: {prompt}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "stream": True
        }

        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    json_data = json.loads(decoded_line[len("data: "):])
                    if "choices" in json_data and len(json_data["choices"]) > 0:
                        chunk = json_data["choices"][0]["delta"].get("content", "")
                        full_response += chunk
                        html_chunk = format_message(full_response)
                        yield html_chunk
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            error_detail = e.response.json()
            yield f"HTTP 400 Error: {error_detail}"
        elif e.response.status_code == 401:
            yield "請輸入正確的 Perplexity API Key"
        else:
            yield f"HTTP Error: {e.response.status_code} - {e.response.reason}"
    except UnicodeEncodeError:
        yield "請輸入正確的 Perplexity API Key"
    except json.JSONDecodeError as e:
        yield f"JSON Decode Error: {str(e)}"
    except Exception as e:
        yield f"Unexpected Error: {str(e)}"

def handle_prompt_submission(prompt):
    if st.session_state['model_type'] == "ChatGPT":
        client = OpenAI(api_key=st.session_state['chatbot_api_key'])
        message_func(prompt, is_user=True)
        thinking_placeholder = st.empty()
        status_text = "Thinking..."
        st.session_state["messages_ChatGPT"].append({"role": "user", "content": prompt})
        st.session_state["messages_ChatGPT"].append({"role": "assistant", "content": status_text})
        messages = st.session_state["messages_ChatGPT"] + [{"role": "user", "content": prompt}]
        with thinking_placeholder.container():
            message_func(status_text, is_user=False)
        response_container = st.empty()
        full_response = ""
        
        for response_message in get_openai_response(
            client,
            st.session_state['open_ai_model'], 
            messages, 
            st.session_state['temperature'], 
            st.session_state['top_p'], 
            st.session_state['presence_penalty'], 
            st.session_state['frequency_penalty'], 
            st.session_state['max_tokens'], 
            st.session_state['gpt_system_prompt'], 
            st.session_state['language']):
            if status_text in [msg['content'] for msg in st.session_state["messages_ChatGPT"] if msg['role'] == 'assistant']:
                st.session_state["messages_ChatGPT"] = [msg for msg in st.session_state["messages_ChatGPT"] if msg['content'] != status_text]
                thinking_placeholder.empty()

            full_response = response_message
            response_container.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: flex-start;">
                    <img src="data:image/png;base64,{assistant_avatar_gpt}" class="bot-avatar" alt="avatar" style="width: 45px; height: 28px;" />
                    <div class="message-container" style="background: #F0F4F8; color: #2B2727; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 5px; margin-left: 5px; font-size: 16px; max-width: 80%; word-wrap: break-word; word-break: break-all;">
                        {full_response} \n </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.session_state["messages_ChatGPT"].append({"role": "assistant", "content": full_response})
        response_container.empty()
        message_func(full_response, is_user=False)
        chat_history_gpt[st.session_state['model_type']] = st.session_state["messages_ChatGPT"]
        save_chat_history(chat_history_gpt, 'ChatGPT')
        st.session_state['prev_response'] = full_response 
        st.rerun()    
        
    elif st.session_state['model_type'] == "Perplexity":
        message_func(prompt, is_user=True)
        prev_state = st.session_state.get('prev_state', {}).get('messages_Perplexity', []).copy()
        thinking_placeholder = st.empty()
        status_text = "Thinking..."
        st.session_state["messages_Perplexity"].append({"role": "user", "content": prompt})
        st.session_state["messages_Perplexity"].append({"role": "assistant", "content": status_text})
        with thinking_placeholder.container():
            message_func(status_text, is_user=False)

        response_container = st.empty()
        full_response = ""
        history = st.session_state["messages_Perplexity"]

        for response_message in generate_perplexity_response(
                prompt,
                history,
                st.session_state['perplexity_model'],
                st.session_state['temperature'],
                st.session_state['top_p'],
                st.session_state['presence_penalty'],
                st.session_state['max_tokens'],
                st.session_state['perplexity_system_prompt'],
                st.session_state['language']):

            if status_text in [msg['content'] for msg in st.session_state["messages_Perplexity"] if msg['role'] == 'assistant']:
                st.session_state["messages_Perplexity"] = [msg for msg in st.session_state["messages_Perplexity"] if msg['content'] != status_text]
                thinking_placeholder.empty()

            full_response = response_message
            response_container.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: flex-start;">
                    <img src="data:image/png;base64,{assistant_avatar_perplexity}" class="bot-avatar" alt="avatar" style="width: 45px; height: 28px;" />
                    <div class="message-container" style="background: #F0F4F8; color: #2B2727; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 5px; margin-left: 5px; font-size: 16px; max-width: 80%; word-wrap: break-word; word-break: break-all;">
                        {full_response} \n </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.session_state["messages_Perplexity"].append({"role": "assistant", "content": full_response})
        response_container.empty()
        message_func(full_response, is_user=False)
        chat_history_perplexity[st.session_state['model_type']] = st.session_state["messages_Perplexity"]
        save_chat_history(chat_history_perplexity, 'Perplexity')
        st.session_state['prev_response'] = full_response  
        if prev_state != st.session_state["messages_Perplexity"]:
            st.session_state['prev_state'] = {'messages_Perplexity': st.session_state["messages_Perplexity"].copy()}

#%% 格式設定與轉換
def format_message(text):
    if isinstance(text, (list, dict)):
        return f"<pre><code>{json.dumps(text, indent=2)}</code></pre>"

    code_pattern = re.compile(r'(```)(.*?)(```|$)', re.DOTALL)
    code_blocks = {}
    code_counter = 0

    def code_replacer(match):
        nonlocal code_counter
        code_key = f"CODE_BLOCK_{code_counter}"
        code_blocks[code_key] = match.group(0)
        code_counter += 1
        return code_key

    text = code_pattern.sub(code_replacer, text)

    header_pattern = re.compile(r'^(#+)\s+(.*)', re.MULTILINE)

    def header_replacer(match):
        content = match.group(2)
        return f"<strong>{content}</strong>"

    text = header_pattern.sub(header_replacer, text)

    bold_pattern = re.compile(r'\*\*(.*?)\*\*')
    text = bold_pattern.sub(r'<b>\1</b>', text)

    tables = parse_markdown_tables(text)
    combined_result = []

    for table in tables:
        if isinstance(table, pd.DataFrame):
            table.index = table.index + 1
            styled_item = table.style.set_table_styles(
                [{'selector': 'thead th', 'props': [('background-color', '#333'), ('color', 'white')]},
                 {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#fffff')]},
                 {'selector': 'tbody tr:hover', 'props': [('background-color', '#DDE3EA')]}]
            ).set_properties(**{'text-align': 'left', 'font-family': 'Arial', 'font-size': '14px'})
            styled_html = styled_item.to_html()
            styled_html = styled_html.replace('border="1"', 'border="0"')
            styled_html = styled_html.replace('<th>', '<th style="border: none;">')
            styled_html = styled_html.replace('<td>', '<td style="border: none;">')
            styled_html = styled_html.replace('<table ', '<table style="width: 100%;" ')
            combined_result.append('<div style="height: 15px;"></div>')
            combined_result.append(f'<div style="display: flex; justify-content: center;">{styled_html}</div>')
        elif isinstance(table, list):
            combined_result.append("<br>".join(map(str, table)))
        else:
            combined_result.append(table)

    combined_text = "\n".join(combined_result)

    for code_key, code_block in code_blocks.items():
        combined_text = combined_text.replace(code_key, code_block)

    return combined_text

def parse_markdown_tables(markdown_text):
    lines = markdown_text.strip().split("\n")
    current_table = []
    combined_result = []

    for line in lines:
        if "|" in line:
            row = [cell.strip() for cell in line.split("|")[1:-1]]
            if not all(cell == '-' * len(cell) for cell in row):
                current_table.append(row)
        else:
            if current_table:
                combined_result.append(current_table)
                current_table = []
            combined_result.append(line)

    if current_table:
        combined_result.append(current_table)

    dfs = []
    for table in combined_result:
        if isinstance(table, list):
            if len(table) > 1:
                header = table[0]
                rows = [row for row in table[1:] if len(row) == len(header)]
                if rows:
                    df = pd.DataFrame(rows, columns=header)
                    dfs.append(df)
                else:
                    dfs.append(table[0])
            else:
                dfs.append(table[0])
        else:
            dfs.append(table)

    return dfs

def message_func(text, is_user=False):
    model_url = f"data:image/png;base64,{assistant_avatar}"
    user_url = f"data:image/png;base64,{st.session_state['user_avatar']}"

    avatar_url = model_url
    if is_user:
        avatar_url = user_url
        message_alignment = "flex-end"
        message_bg_color = "linear-gradient(135deg, #30A2FD 0%, #035DE5 100%)"
        avatar_class = "user-avatar"
        avatar_size = "width: 32px; height: 40px;"
        text_with_line_breaks = html.escape(text).replace("\n", "<br>")
        st.markdown(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: {message_alignment};">
                    <div class="message-container" style="background: {message_bg_color}; color: white; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 10px; font-size: 16px; max-width: 80%; word-wrap: break-word; word-break: break-all;">
                        {text_with_line_breaks} \n </div>
                    <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="{avatar_size}" />
                </div>
                """,
            unsafe_allow_html=True,
        )
    else:
        message_alignment = "flex-start"
        message_bg_color = "#F0F4F8"
        avatar_class = "bot-avatar"
        avatar_size = "width: 45px; height: 28px;"
        if assistant_avatar == assistant_avatar_perplexity:
            avatar_class += " perplexity"

        result = parse_markdown_tables(text)
        combined_result = []

        for item in result:
            if isinstance(item, pd.DataFrame):
                item.index = item.index + 1
                styled_item = item.style.set_table_styles(
                    [{'selector': 'thead th', 'props': [('background-color', '#333'), ('color', 'white')]},
                     {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]},
                     {'selector': 'tbody tr:hover', 'props': [('background-color', '#ddd')]}]
                ).set_properties(**{'text-align': 'left', 'font-family': 'Arial', 'font-size': '14px'})
                combined_result.append('<div style="height: 15px;"></div>')  
                styled_html = styled_item.to_html()
                styled_html = styled_html.replace('border="1"', 'border="0"')  
                styled_html = styled_html.replace('<th>', '<th style="border: none;">') 
                styled_html = styled_html.replace('<td>', '<td style="border: none;">') 
                combined_result.append(f'<div style="display: flex; justify-content: center;">{styled_html}</div>')
            else:
                combined_result.append(item)

        combined_text = "\n".join(combined_result)
        formatted_message = format_message(combined_text)

        st.markdown(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: {message_alignment};">
                    <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="{avatar_size}" />
                    <div class="message-container" style="background: {message_bg_color}; color: #2B2727; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 5px; margin-left: 5px; font-size: 16px; max-width: 80%; word-wrap: break-word; word-break: break-all;">
                        {formatted_message} \n </div>
                </div>
                """,
            unsafe_allow_html=True,
        )
        
#%% 側邊欄設置
def update_openai_api_key():
    if st.session_state['openai_api_key_input'] != st.session_state['chatbot_api_key']:
        st.session_state['chatbot_api_key'] = st.session_state['openai_api_key_input']
        settings['chatbot_api_key'] = st.session_state['chatbot_api_key']
        save_settings(settings)
        if not st.session_state['gpt_chat_started']:
            if st.session_state["messages_ChatGPT"]:
                st.session_state["messages_ChatGPT"][0]['content'] = []

def update_perplexity_api_key():
    if st.session_state['perplexity_api_key_input'] != st.session_state['perplexity_api_key']:
        st.session_state['perplexity_api_key'] = st.session_state['perplexity_api_key_input']
        settings['perplexity_api_key'] = st.session_state['perplexity_api_key']
        save_settings(settings)
        if not st.session_state['perplexity_chat_started']:
            if st.session_state["messages_Perplexity"]:
                st.session_state["messages_Perplexity"][0]['content'] = []


def reset_chat(confirm_reset=None):
    if confirm_reset is None:
        st.session_state['reset_confirmation'] = True
    else:
        if confirm_reset:
            if st.session_state['model_type'] == 'ChatGPT':
                st.session_state['gpt_chat_started'] = False
                st.session_state['messages_ChatGPT'] = []
                if os.path.exists('chat_history_gpt.json'):
                    os.remove('chat_history_gpt.json')
            elif st.session_state['model_type'] == 'Perplexity':
                st.session_state['perplexity_chat_started'] = False
                st.session_state['messages_Perplexity'] = []
                if os.path.exists('chat_history_perplexity.json'):
                    os.remove('chat_history_perplexity.json')
            elif st.session_state['model_type'] == 'DALL-E':
                st.session_state['dalle_chat_started'] = False
                st.session_state['messages_DALLE'] = []
                if os.path.exists('chat_history_dalle.json'):
                    os.remove('chat_history_dalle.json')

            placeholder_status = {
                'ChatGPT': st.session_state['gpt_chat_started'],
                'Perplexity': st.session_state['perplexity_chat_started'],
                'DALL-E': st.session_state['dalle_chat_started']
            }
            with open('text_placeholder_status.json', 'w') as f:
                json.dump(placeholder_status, f)
            
            st.session_state['reset_confirmation'] = False
        else:
            st.session_state['reset_confirmation'] = False

def cancel_reset_chat():
    st.session_state['reset_confirmation'] = False

def confirm_reset_chat(confirm_reset=None):
    if confirm_reset is None:
        confirm, cancel = st.columns(2)
        with confirm:
            st.button("確認", key="confirm_reset", on_click=confirm_reset_chat, args=(True,))
        with cancel:
            st.button("取消", key="cancel_reset", on_click=confirm_reset_chat, args=(False,))
    else:
        st.session_state['reset_confirmed'] = confirm_reset
        if st.session_state.get('reset_confirmed'):
            reset_chat()
        st.session_state['reset_confirmation'] = False
        
with st.sidebar:
    selected = option_menu("",
        ["對話",'AI生圖','模型設定','提示詞','頭像'],
        icons=['chat-dots-fill','palette-fill','gear-fill','info-square-fill','person-square'], menu_icon="robot", default_index=0,
        styles={
            "container": {"padding": "0!important", "background": "#F0F4F8"},
            "icon": {"padding": "0px 10px 0px 0px !important","color": "#46474A", "font-size": "17px"},
            "nav-link": {"padding": "7px 0px 7px 15px","font-size": "17px", "text-align": "left", "margin":"3px", "--hover-color": "#E9EEF6","border-radius": "20px"},
            "nav-link-selected": {"padding": "7px 0px 7px 15px","background": "#B4D7FF", "color": "#041E49","border-radius": "20px"},
        }
    )
    
    if selected in ["對話", "模型設定"]:
        model_toggle = st.selectbox("",options=["ChatGPT", "Perplexity"], key="model_type", label_visibility="collapsed")
        st.write("\n")
    elif selected == "AI生圖":
        model_toggle = st.selectbox("",options=["DALL ·E 3", "DALL ·E 2"], key="dalle_model_display", label_visibility="collapsed")
        st.write("\n")
    

    if selected =="對話":
        if st.session_state["model_type"] == "Perplexity":
            assistant_avatar = assistant_avatar_perplexity
            perplexity_api_key_input = st.text_input("請輸入 Perplexity API Key", value=st.session_state.get('perplexity_api_key', ''), type="password", key='perplexity_api_key_input', on_change=update_perplexity_api_key)
        
        elif st.session_state["model_type"] in ["ChatGPT", "DALL ·E 3", "DALL ·E 2"]:
            assistant_avatar = assistant_avatar_gpt
            openai_api_key_input = st.text_input("請輸入 OpenAI API Key", value=st.session_state.get('chatbot_api_key', ''), type="password", key='openai_api_key_input', on_change=update_openai_api_key)
    
    if selected == "AI生圖":
        assistant_avatar = assistant_avatar_gpt
        openai_api_key_input = st.text_input("請輸入 OpenAI API Key", value=st.session_state.get('chatbot_api_key', ''), type="password", key='openai_api_key_input', on_change=update_openai_api_key)
        
    if selected == "AI生圖":
        dalle_model_map = {"DALL ·E 3": "dall-e-3", "DALL ·E 2": "dall-e-2"}
        st.session_state["dalle_model"] = dalle_model_map[st.session_state["dalle_model_display"]]

#%% 對話頁面
def update_gpt_system_prompt():
    st.session_state['gpt_system_prompt'] = st.session_state['gpt_system_prompt_input']
    save_settings({
        'gpt_system_prompt': st.session_state['gpt_system_prompt']
    })

def update_perplexity_system_prompt():
    st.session_state['perplexity_system_prompt'] = st.session_state['perplexity_system_prompt_input']
    save_settings({
        'perplexity_system_prompt': st.session_state['perplexity_system_prompt']
    })
    
def update_exported_shortcuts():
    for exported_shortcut in st.session_state.get('exported_shortcuts', []):
        for shortcut in st.session_state['shortcuts']:
            if exported_shortcut['name'] == shortcut['name']:
                exported_shortcut.update(shortcut)
                
def hide_expander():
    st.session_state['expander_state'] = False
    st.session_state['active_shortcut'] = None

def greeting_based_on_time():
    tz = pytz.timezone('Asia/Taipei')
    now = datetime.now(tz)
    current_hour = now.hour
    if 5 <= current_hour < 12:
        return "早安，你好！"
    elif 12 <= current_hour < 18:
        return "午安，你好！"
    else:
        return "晚安，你好！"
greeting_message = greeting_based_on_time()

def render_initial_placeholder():
    html_code = f"""
        <style>
            .gradient-text {{
                font-size: 55px;
                font-weight:bold;
                background: linear-gradient(to right, #5282ED, #9874CE, #D96470);
                -webkit-background-clip: text;
                color: transparent;
                display: inline;
            }}
            .text {{
                font-size: 50px;
                font-weight:bold;
                background: #C5C7C5;
                -webkit-background-clip: text;
                color: transparent;
                display: inline;
            }}
        </style>
        <div class="gradient-text">{greeting_message}</div>
        <br> 
        <div class="text">有什麼我可以幫上忙的嗎？</div>
        """
    if st.session_state['text_placeholder']:
        st.session_state['text_placeholder'].markdown(html_code, unsafe_allow_html=True)

if selected == "對話" and 'exported_shortcuts' in st.session_state:
    api_key_entered = (st.session_state['model_type'] == "ChatGPT" and st.session_state['chatbot_api_key']) or \
              (st.session_state['model_type'] == "Perplexity" and st.session_state['perplexity_api_key'])

    if api_key_entered and 'exported_shortcuts' in st.session_state and not (st.session_state['model_type'] == "ChatGPT" and st.session_state['open_ai_model'] == "DALL-E"):
        with st.sidebar.expander('你的提示詞'):
            for idx, shortcut in enumerate(st.session_state['exported_shortcuts']):
                if shortcut['target'] == 'chat':  
                    col = st.columns(1)[0]
                    with col:
                        if ui.button(shortcut['name'], key=f'exported_shortcut_{idx}', style={"width": "100%", "background": "#C4DDA7", "color": "#2b2727"}):
                            st.session_state['active_shortcut'] = shortcut

    if 'exported_shortcuts' in st.session_state and not (st.session_state['model_type'] == "ChatGPT" and st.session_state['open_ai_model'] == "DALL-E"):
        with st.sidebar:
            st.divider()
            if st.session_state.get('reset_confirmation'):
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    st.button("確認", on_click=reset_chat, args=(True,))
                with cancel_col:
                    st.button("取消", on_click=reset_chat, args=(False,))
            else:
                st.button("重置對話", on_click=reset_chat)

    if selected == "對話":
        if st.session_state['reset_confirmed']:
            st.session_state['reset_confirmed'] = False
            
        if f"messages_{st.session_state['model_type']}" not in st.session_state:
            st.session_state[f"messages_{st.session_state['model_type']}"] = []
        
        if not (st.session_state['model_type'] == "ChatGPT" and st.session_state['open_ai_model'] == "DALL-E"):
            for msg in st.session_state[f"messages_{st.session_state['model_type']}"]:
                message_func(msg["content"], is_user=(msg["role"] == "user"))
    
        if st.session_state['model_type'] == "ChatGPT":
            if not st.session_state['open_ai_model'] == "DALL-E":
                if st.session_state['gpt_chat_started'] == False and not st.session_state['prompt_submitted']:
                    st.session_state['text_placeholder'] = st.empty()
                    render_initial_placeholder()

                prompt = st.chat_input()
                if prompt:
                    if not st.session_state['chatbot_api_key']:
                        message_func("請輸入您的 OpenAI API Key", is_user=False)
                    else:
                        if st.session_state['gpt_chat_started'] == False:
                            st.session_state['gpt_chat_started'] = True
                        if st.session_state['text_placeholder']:
                            st.session_state['text_placeholder'].empty()
                            st.session_state['text_placeholder'] = None 

                        handle_prompt_submission(prompt)
    
        if st.session_state['model_type'] == "Perplexity" and not st.session_state['prompt_submitted']:
            if st.session_state['perplexity_chat_started'] == False :
                st.session_state['text_placeholder'] = st.empty()
                render_initial_placeholder()
                
            prompt = st.chat_input()
            if prompt:
                if not st.session_state['perplexity_api_key']:
                    message_func("請輸入您的 Perplexity API Key", is_user=False)
                else:
                    if st.session_state['perplexity_chat_started'] == False:
                        st.session_state['perplexity_chat_started'] = True
                    if st.session_state['text_placeholder']:
                        st.session_state['text_placeholder'].empty()
                        st.session_state['text_placeholder'] = None 
                        
                    handle_prompt_submission(prompt)

    if 'active_shortcut' in st.session_state and st.session_state.get('active_shortcut') is not None and st.session_state['active_shortcut']['target'] == 'chat':
        shortcut = st.session_state['active_shortcut']
        inputs = {}
        form_placeholder = st.empty()
        with form_placeholder.form(key=f'prompt_template_form_{shortcut["name"]}'):
            col1, col2 = st.columns(2)
            for i, component in enumerate(shortcut['components']):
                with col1 if i % 2 == 0 else col2:
                    if component['type'] == "text input":
                        inputs[component['label']] = st.text_input(component['label'], key=f'shortcut_text_input_{i}')
                    elif component['type'] == "selector":
                        inputs[component['label']] = st.selectbox(component['label'], component['options'], key=f'shortcut_selector_{i}')
                    elif component['type'] == "multi selector":
                        inputs[component['label']] = st.multiselect(component['label'], component['options'], key=f'shortcut_multi_selector_{i}')
    
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("取消", on_click=hide_expander):
                    st.session_state['active_shortcut'] = None
                    form_placeholder.empty()
    
            with col2:
                提示詞模板 = st.form_submit_button("送出")
    
        if 提示詞模板 and not st.session_state['prompt_submitted']:
            if st.session_state['model_type'] == "ChatGPT":
                if st.session_state['gpt_chat_started'] == False:
                    st.session_state['gpt_chat_started'] = True
                if st.session_state['text_placeholder']:
                    st.session_state['text_placeholder'].empty()
                    st.session_state['text_placeholder'] = None 
            elif st.session_state['model_type'] == "Perplexity":
                if st.session_state['gpt_chat_started'] == False:
                    st.session_state['perplexity_chat_started'] = True
                if st.session_state['text_placeholder']:
                    st.session_state['text_placeholder'].empty()
                    st.session_state['text_placeholder'] = None 
                    
            st.session_state['active_shortcut'] = None
            st.session_state['expander_state'] = False
            form_placeholder.empty()
            prompt_template = shortcut['prompt_template'].replace("{", "{{").replace("}", "}}")
            for key in inputs.keys():
                prompt_template = prompt_template.replace(f"{{{{{key}}}}}", f"{inputs[key]}")
            try:
                prompt = prompt_template.replace("{{", "{").replace("}}", "}")  
                handle_prompt_submission(prompt)
                st.rerun()
                st.session_state['prompt_submitted'] = True
            except KeyError as e:
                st.error(f"缺少必需的輸入: {e}")

    if 'prompt_submitted' in st.session_state:
        del st.session_state['prompt_submitted']

#%% 模型設定頁面
def update_open_ai_model():
    model_display_names = {"GPT-4o": "gpt-4o", "GPT-4o mini": "gpt-4o-mini"}
    selected_model = model_display_names[st.session_state['open_ai_model_selection']]
    st.session_state['open_ai_model'] = selected_model
    settings['open_ai_model'] = selected_model
    save_settings(settings)
    st.session_state['update_trigger'] = not st.session_state.get('update_trigger', False)

def update_perplexity_model():
    perplexity_model_display_names = {
        "Sonar-Large 128k Online": "llama-3.1-sonar-large-128k-online",
        "Sonar-Large 128k Chat": "llama-3.1-sonar-large-128k-chat",
    }
    selected_model = perplexity_model_display_names[st.session_state['perplexity_model_selection']]
    st.session_state['perplexity_model'] = selected_model
    settings['perplexity_model'] = selected_model
    save_settings(settings)
    st.session_state['update_trigger'] = not st.session_state.get('update_trigger', False)
    
def update_model_params(model_type):
    if model_type == "ChatGPT":
        st.session_state['temperature'] = st.session_state['temperature_slider']
        st.session_state['top_p'] = st.session_state['top_p_slider']
        st.session_state['presence_penalty'] = st.session_state['presence_penalty_slider']
        st.session_state['frequency_penalty'] = st.session_state['frequency_penalty_slider']
        save_settings({
            'temperature': st.session_state['temperature'],
            'top_p': st.session_state['top_p'],
            'presence_penalty': st.session_state['presence_penalty'],
            'frequency_penalty': st.session_state['frequency_penalty']
        })
    elif model_type == "Perplexity":
        st.session_state['perplexity_temperature'] = st.session_state['perplexity_temperature_slider']
        st.session_state['perplexity_top_p'] = st.session_state['perplexity_top_p_slider']
        st.session_state['perplexity_presence_penalty'] = st.session_state['perplexity_presence_penalty_slider']
        save_settings({
            'perplexity_temperature': st.session_state['perplexity_temperature'],
            'perplexity_top_p': st.session_state['perplexity_top_p'],
            'perplexity_presence_penalty': st.session_state['perplexity_presence_penalty']
        })

def update_language():
    if st.session_state['language_input'] != st.session_state['language']:
        st.session_state['language'] = st.session_state['language_input']
        save_settings({
            'language': st.session_state['language']
        })

def update_max_tokens():
    if st.session_state['max_tokens_input'] != st.session_state['max_tokens']:
        st.session_state['max_tokens'] = st.session_state['max_tokens_input']
        save_settings({
            'max_tokens': st.session_state['max_tokens']
        })

def update_slider(key, value):
    st.session_state[key] = value
    save_settings({
        'temperature': st.session_state['temperature'],
        'top_p': st.session_state['top_p'],
        'presence_penalty': st.session_state['presence_penalty'],
        'frequency_penalty': st.session_state['frequency_penalty']
    })

if selected == "模型設定":
    col1, col2, col3 = st.columns([2, 2, 1.5])
    if st.session_state['model_type'] == "ChatGPT":
        with col1:
            model_display_names = {"GPT-4o": "gpt-4o", "GPT-4o mini": "gpt-4o-mini"}
            reverse_mapping = {v: k for k, v in model_display_names.items()}
            selected_model_key = reverse_mapping.get(st.session_state['open_ai_model'], "GPT-4o")
            st.selectbox(
                "選擇 ChatGPT 模型",
                list(model_display_names.keys()),
                index=list(model_display_names.keys()).index(selected_model_key),
                key='open_ai_model_selection',
                on_change=update_open_ai_model
            )
        with col2:
            st.text_input(
                "指定使用的語言",
                key='language_input',
                value=st.session_state.get('language', ''),
                on_change=update_language
            )
        with col3:
            st.number_input(
                "Tokens 上限",
                key='max_tokens_input',
                min_value=0,
                value=st.session_state.get('max_tokens', 1000),
                on_change=update_max_tokens
            )
      
        st.text_area(
            "角色設定",
            value=st.session_state.get('gpt_system_prompt', ''),
            placeholder="你是一個友好且資深的英文老師。你的目標是幫助使用者提高他們的語言能力，並且用簡單易懂的方式解釋概念。你應該耐心回答問題，並鼓勵學生提出更多問題。",
            key="gpt_system_prompt_input",
            on_change=update_gpt_system_prompt,
            height=290
        )

        with st.expander("模型參數", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.slider(
                    "選擇 Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    step=0.1,
                    value=st.session_state['temperature'],
                    key='temperature_slider',
                    on_change=update_model_params,
                    args=("ChatGPT",)
                )
                st.slider(
                    "選擇 Presence Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    step=0.1,
                    value=st.session_state['presence_penalty'],
                    key='presence_penalty_slider',
                    on_change=update_model_params,
                    args=("ChatGPT",)
                )
            with col2:
                st.slider(
                    "選擇 Top P",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    value=st.session_state['top_p'],
                    key='top_p_slider',
                    on_change=update_model_params,
                    args=("ChatGPT",)
                )
                st.slider(
                    "選擇 Frequency Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    step=0.1,
                    value=st.session_state['frequency_penalty'],
                    key='frequency_penalty_slider',
                    on_change=update_model_params,
                    args=("ChatGPT",)
                )

    elif st.session_state['model_type'] == "Perplexity":
        with col1:
            perplexity_model_display_names = {
                "Sonar-Large 128k Online": "llama-3.1-sonar-large-128k-online",
                "Sonar-Large 128k Chat": "llama-3.1-sonar-large-128k-chat",
            }
            reverse_mapping = {v: k for k, v in perplexity_model_display_names.items()}
            selected_model_key = reverse_mapping.get(st.session_state['perplexity_model'], "Sonar-Large 128k Online")
            st.selectbox(
                "選擇 Sonar 或 Llama3 模型",
                list(perplexity_model_display_names.keys()),
                index=list(perplexity_model_display_names.keys()).index(selected_model_key),
                key='perplexity_model_selection',
                on_change=update_perplexity_model
            )
        with col2:
            st.text_input(
                "指定使用的語言",
                key='language_input',
                value=st.session_state.get('language', ''),
                on_change=update_language
            )
        with col3:
            st.number_input(
                "Tokens 上限",
                key='max_tokens_input',
                min_value=0,
                value=st.session_state.get('max_tokens', 1000),
                on_change=update_max_tokens
            )
  
        st.text_area(
            "角色設定",
            value=st.session_state.get('perplexity_system_prompt', ''),
            placeholder="你是一個專業的科技支援工程師。你的目標是幫助用戶解決各種技術問題，無論是硬體還是軟體問題。你應該詳細解釋解決方案，並確保用戶理解每一步驟。",
            key="perplexity_system_prompt_input",
            on_change=update_perplexity_system_prompt,
            height=290
        )
        with st.expander("模型參數", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.slider(
                    "選擇 Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    step=0.1,
                    value=st.session_state['perplexity_temperature'],
                    key='perplexity_temperature_slider',
                    on_change=update_model_params,
                    args=("Perplexity",)
                )
            with col2:
                st.slider(
                    "選擇 Top P",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    value=st.session_state['perplexity_top_p'],
                    key='perplexity_top_p_slider',
                    on_change=update_model_params,
                    args=("Perplexity",)
                )

            st.slider(
                "選擇 Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                step=0.1,
                value=st.session_state['perplexity_presence_penalty'],
                key='perplexity_presence_penalty_slider',
                on_change=update_model_params,
                args=("Perplexity",)
            )

    settings['open_ai_model'] = st.session_state['open_ai_model']
    settings['perplexity_model'] = st.session_state['perplexity_model']
    settings['perplexity_temperature'] = st.session_state['perplexity_temperature']
    settings['perplexity_top_p'] = st.session_state['perplexity_top_p']
    settings['perplexity_presence_penalty'] = st.session_state['perplexity_presence_penalty']
    settings['perplexity_max_tokens'] = st.session_state['max_tokens']
    settings['perplexity_system_prompt'] = st.session_state['perplexity_system_prompt']
    settings['gpt_system_prompt'] = st.session_state['gpt_system_prompt']
    settings['language'] = st.session_state['language']
    settings['temperature'] = st.session_state['temperature']
    settings['top_p'] = st.session_state['top_p']
    settings['presence_penalty'] = st.session_state['presence_penalty']
    settings['frequency_penalty'] = st.session_state['frequency_penalty']
    settings['max_tokens'] = st.session_state['max_tokens']
    save_settings(settings)

#%% AI生圖頁面
def render_ai_image_placeholder():
    """渲染AI生圖頁面的歡迎文字"""
    html_code = f"""
        <style>
            .gradient-text {{
                font-size: 55px;
                font-weight:bold;
                background: linear-gradient(to right, #5282ED, #9874CE, #D96470);
                -webkit-background-clip: text;
                color: transparent;
                display: inline;
            }}
            .text {{
                font-size: 50px;
                font-weight:bold;
                background: #C5C7C5;
                -webkit-background-clip: text;
                color: transparent;
                display: inline;
            }}
        </style>
        <div class="gradient-text">{greeting_message}</div>
        <br> <!-- 加入換行 -->
        <div class="text">今天想創造些什麼？</div>
        """
    st.session_state['text_placeholder'].markdown(html_code, unsafe_allow_html=True)

def update(action):
    if action == 'reset':
        st.session_state['reset_confirmation'] = True
    elif action == 'confirm':
        st.session_state["messages_DALLE"] = []
        st.session_state['dalle_chat_started'] = False
        chat_history_dalle['DALL-E'] = st.session_state["messages_DALLE"]
        save_chat_history(chat_history_dalle, 'DALL-E')
        st.session_state['reset_confirmation'] = False
    elif action == 'cancel':
        st.session_state['reset_confirmation'] = False

def generate_image_from_prompt(prompt, model):
    client = OpenAI(api_key=st.session_state['chatbot_api_key'])

    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
            n=1
        )
        image_url = response.data[0].url

        response = requests.get(image_url)

        img_base64 = base64.b64encode(BytesIO(response.content).getvalue()).decode("utf-8")

        return img_base64

    except Exception as e:
        st.error(f"圖片生成失敗：{str(e)}")
        return None
    
def handle_image_generation(prompt):
    img_base64 = generate_image_from_prompt(prompt, st.session_state['dalle_model'])
    if img_base64:
        thinking_placeholder.empty()
        st.session_state["messages_DALLE"] = [msg for msg in st.session_state["messages_DALLE"] if msg["content"] != status_text]
        st.session_state["messages_DALLE"].append({"role": "assistant", "content": f'<img src="data:image/png;base64,{img_base64}" alt="Generated Image" style="max-width: 100%;">'})
        response_container.empty()
        message_func(f'<img src="data:image/png;base64,{img_base64}" alt="Generated Image" style="max-width: 100%;">', is_user=False)
        chat_history_dalle['DALL-E'] = st.session_state["messages_DALLE"]
        save_chat_history(chat_history_dalle, 'DALL-E')

if selected == "AI生圖":
    if 'messages_DALLE' not in st.session_state:
        st.session_state['messages_DALLE'] = chat_history_dalle.get('DALL-E', [])
    if st.session_state['dalle_chat_started'] == False and not st.session_state['prompt_submitted']:
        st.session_state['text_placeholder'] = st.empty()
        render_ai_image_placeholder()

    for msg in st.session_state["messages_DALLE"]:
        message_func(msg["content"], is_user=(msg["role"] == "user"))
    
    if 'exported_shortcuts' in st.session_state:
        with st.sidebar.expander('你的提示詞', expanded=False):
            for idx, shortcut in enumerate(st.session_state['exported_shortcuts']):
                if shortcut['target'] == 'image':  
                    col = st.columns(1)[0]
                    with col:
                        if ui.button(shortcut['name'], key=f'exported_shortcut_{idx}', style={"width": "100%", "background": "#B4D7FF", "color": "#2b2727"}):
                            st.session_state['active_shortcut'] = shortcut
    
    prompt = st.chat_input()
    if prompt:
        if not st.session_state['chatbot_api_key']:
            message_func("請輸入您的 OpenAI API Key", is_user=False)
        else:
            st.session_state['dalle_chat_started'] = True
            if st.session_state['text_placeholder']:
                st.session_state['text_placeholder'].empty()
                st.session_state['text_placeholder'] = None  
            st.session_state["messages_DALLE"].append({"role": "user", "content": prompt})
            message_func(prompt, is_user=True)
    
            thinking_placeholder = st.empty()
            status_text = "圖片生成中..."
            st.session_state["messages_DALLE"].append({"role": "assistant", "content": status_text})
            with thinking_placeholder.container():
                message_func(status_text, is_user=False)
    
            response_container = st.empty()
            handle_image_generation(prompt)
            st.session_state['prompt_submitted'] = True

    if 'active_shortcut' in st.session_state and st.session_state.get('active_shortcut') is not None and st.session_state['active_shortcut']['target'] == 'image':
        shortcut = st.session_state['active_shortcut']
        inputs = {}
        form_placeholder = st.empty()
        with form_placeholder.form(key=f'prompt_template_form_{shortcut["name"]}'):
            col1, col2 = st.columns(2)
            for i, component in enumerate(shortcut['components']):
                with col1 if i % 2 == 0 else col2:
                    if component['type'] == "text input":
                        inputs[component['label']] = st.text_input(component['label'], key=f'shortcut_text_input_{i}')
                    elif component['type'] == "selector":
                        inputs[component['label']] = st.selectbox(component['label'], component['options'], key=f'shortcut_selector_{i}')
                    elif component['type'] == "multi selector":
                        inputs[component['label']] = st.multiselect(component['label'], component['options'], key=f'shortcut_multi_selector_{i}')
    
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("取消", on_click=hide_expander):
                    st.session_state['active_shortcut'] = None
                    form_placeholder.empty()
    
            with col2:
                提示詞模板 = st.form_submit_button("送出")
    
        if 提示詞模板 and not st.session_state.get('prompt_submitted', False):
            st.session_state['dalle_chat_started'] = True
            if st.session_state['text_placeholder']:
                st.session_state['text_placeholder'].empty()
                st.session_state['text_placeholder'] = None 
            st.session_state['active_shortcut'] = None
            st.session_state['expander_state'] = False
            form_placeholder.empty()
            prompt_template = shortcut['prompt_template'].replace("{", "{{").replace("}", "}}")
            for key in inputs.keys():
                prompt_template = prompt_template.replace(f"{{{{{key}}}}}", f"{inputs[key]}")
            try:
                prompt = prompt_template.replace("{{", "{").replace("}}", "}")
                st.session_state["messages_DALLE"].append({"role": "user", "content": prompt})
                message_func(prompt, is_user=True)
                thinking_placeholder = st.empty()
                status_text = "圖片生成中..."
                st.session_state["messages_DALLE"].append({"role": "assistant", "content": status_text})
                with thinking_placeholder.container():
                    message_func(status_text, is_user=False)
    
                response_container = st.empty()
                handle_image_generation(prompt)
                st.rerun()
                st.session_state['prompt_submitted'] = True
            except KeyError as e:
                st.error(f"缺少必需的輸入: {e}")
    
    if 'prompt_submitted' in st.session_state:
        del st.session_state['prompt_submitted']
                
    with st.sidebar:
        st.divider()
        if 'reset_confirmation' not in st.session_state:
            st.session_state['reset_confirmation'] = False
        if st.session_state['reset_confirmation']:
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                st.button("確認", key="confirm_reset_dalle", on_click=lambda: update('confirm'))
            with cancel_col:
                st.button("取消", key="cancel_reset_dalle", on_click=lambda: update('cancel'))
        else:
            st.button("重置對話", key="reset_chat_dalle", on_click=lambda: update('reset'))
            
#%% 提示詞頁面
def reset_new_component():
    st.session_state['new_component'] = {"label": "", "options": ""}

def add_shortcut():
    if len(st.session_state['shortcuts']) >= 8:
        st.markdown("<div class='custom-warning'>已達到 Shortcut 數量上限（8 個）</div>", unsafe_allow_html=True)
        time.sleep(1)
        st.rerun()
    else:
        existing_names = [shortcut['name'] for shortcut in st.session_state['shortcuts']]
        base_name = f"Shortcut {len(st.session_state['shortcuts']) + 1}"
        new_name = base_name
        counter = 1
        while new_name in existing_names:
            counter += 1
            new_name = f"{base_name} ({counter})"

        new_shortcut = {
            "name": new_name,
            "components": [],
            "prompt_template": ""
        }
        st.session_state['shortcuts'].append(new_shortcut)
        st.session_state['shortcut_names'].append(new_shortcut["name"])
        st.session_state['current_shortcut'] = len(st.session_state['shortcuts']) - 1
        save_shortcuts()

def delete_shortcut(index):
    if len(st.session_state['shortcuts']) > 0:
        deleted_shortcut = st.session_state['shortcuts'].pop(index)
        st.session_state['shortcut_names'].pop(index)
        st.session_state['current_shortcut'] = max(0, index - 1)
        
        st.session_state['exported_shortcuts'] = [
            shortcut for shortcut in st.session_state['exported_shortcuts']
            if shortcut['name'] != deleted_shortcut['name']
        ]
        
        del st.session_state[f'prompt_template_{index}']
        for i, component in enumerate(deleted_shortcut['components']):
            if component['type'] == "text input":
                del st.session_state[f'text_input_{index}_{i}']
            elif component['type'] == "selector":
                del st.session_state[f'selector_{index}_{i}']
            elif component['type'] == "multi selector":
                del st.session_state[f'multi_selector_{index}_{i}']
        
        if len(st.session_state['shortcuts']) == 0:
            add_shortcut() 
        
        save_shortcuts()
        st.session_state['update_trigger'] = not st.session_state.get('update_trigger', False)

def confirm_delete_shortcut(index=None, confirm_delete=None):
    if confirm_delete is None:
        st.session_state['delete_confirmation'] = index
    else:
        if confirm_delete:
            delete_shortcut(index)
            st.session_state['delete_confirmation'] = None
        else:
            st.session_state['delete_confirmation'] = None
            
def cancel_delete_shortcut():
    st.session_state['delete_confirmation'] = None
    
def update_prompt_template(idx):
    st.session_state['shortcuts'][idx]['prompt_template'] = st.session_state[f'prompt_template_{idx}']
    update_exported_shortcuts()
    save_shortcuts()

def update_shortcut_name(idx):
    new_name = st.session_state[f'shortcut_name_{idx}']
    if new_name != st.session_state['shortcuts'][idx]['name']:
        old_name = st.session_state['shortcuts'][idx]['name']
        st.session_state['shortcuts'][idx]['name'] = new_name
        for exported_shortcut in st.session_state['exported_shortcuts']:
            if exported_shortcut['name'] == old_name:
                exported_shortcut['name'] = new_name
        save_shortcuts()
        st.session_state['update_trigger'] = not st.session_state.get('update_trigger', False)

if selected == "提示詞":
    with st.sidebar:
        st.divider()
        if st.button("新增提示詞"):
            add_shortcut()

    if st.session_state['shortcuts']:
        tabs = st.tabs([shortcut['name'] for shortcut in st.session_state['shortcuts']])
        for idx, tab in enumerate(tabs):
            with tab:
                st.session_state['current_shortcut'] = idx
                shortcut = st.session_state['shortcuts'][idx]

                if f'prompt_template_{idx}' not in st.session_state:
                    st.session_state[f'prompt_template_{idx}'] = shortcut['prompt_template']

                col1, col2 = st.columns([1, 2.5])
                with col1:
                    component_type = st.selectbox("選擇要新增的元件類型", ["文字輸入", "選單", "多選選單"], key=f'component_type_{idx}')
                with col2:
                    new_name = st.text_input("提示詞名稱", value=shortcut['name'], key=f'shortcut_name_{idx}', on_change=update_shortcut_name, args=(idx,))
                    if new_name.strip() == "":
                        st.warning("名稱不能為空")

                if component_type == "文字輸入":
                    with st.expander("建立文字變數", expanded=True):
                        label = st.text_input("變數名稱", key=f'text_input_label_{idx}')
                        col1,col2 = st.columns(2)
                        if col1.button("新增 文字輸入", key=f'add_text_input_{idx}'):
                            if label:
                                shortcut['components'].append({"type": "text input", "label": label})
                                reset_new_component()
                                update_exported_shortcuts()
                                save_shortcuts()
                                st.success("已成功新增")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning("標籤為必填項目")
                                time.sleep(1)
                                st.rerun()

                elif component_type == "選單":
                    with st.expander("建立選單變數", expanded=True):
                        label = st.text_input("變數名稱", key=f'selector_label_{idx}')
                        options = st.text_area("輸入選項（每行一個）", key=f'selector_options_{idx}').split("\n")
                        col1,col2 = st.columns(2)
                        if col1.button("新增 選單", key=f'add_selector_{idx}'):
                            if label and options and all(option.strip() for option in options):
                                shortcut['components'].append({"type": "selector", "label": label, "options": options})
                                reset_new_component()
                                update_exported_shortcuts()
                                save_shortcuts()
                                st.success("已成功新增")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning("標籤和選項為必填項目")
                                time.sleep(1)
                                st.rerun()

                elif component_type == "多選選單":
                    with st.expander("建立多選選單變數", expanded=True):
                        label = st.text_input("變數名稱", key=f'multi_selector_label_{idx}')
                        options = st.text_area("輸入選項（每行一個）", key=f'multi_selector_options_{idx}').split("\n")
                        col1,col2 = st.columns(2)
                        if col1.button("新增 多選選單", key=f'add_multi_selector_{idx}'):
                            if label and options and all(option.strip() for option in options):
                                shortcut['components'].append({"type": "multi selector", "label": label, "options": options})
                                reset_new_component()
                                update_exported_shortcuts()
                                save_shortcuts()
                                st.success("已成功新增")
                                time.sleep(1)
                                st.rerun()

                with st.expander("你的元件組合",expanded=False):
                    cols = st.columns(4)
                    for i, component in enumerate(shortcut['components']):
                        col = cols[i % 4]
                        with col:
                            if component['type'] == "text input":
                                st.text_input(component['label'], key=f'text_input_{idx}_{i}')
                            elif component['type'] == "selector":
                                st.selectbox(component['label'], component['options'], key=f'selector_{idx}_{i}')
                            elif component['type'] == "multi selector":
                                st.multiselect(component['label'], component['options'], key=f'multi_selector_{idx}_{i}')
                            if st.button("刪除", key=f'delete_{idx}_{i}'):
                                del shortcut['components'][i]
                                update_exported_shortcuts()
                                save_shortcuts()
                                st.rerun()

                st.divider()
                st.write("##### 自訂提示詞公式")
                st.write("\n")
                col1, col2, col3 = st.columns([2, 0.1, 1.8])
                with col1:
                    st.text_area("", height=350, placeholder="用{ }代表標籤變數", key=f'prompt_template_{idx}', label_visibility="collapsed", on_change=update_prompt_template, args=(idx,))
                with col3:
                    st.write("##### 提示詞預覽")
                    inputs = {}
                    for i, component in enumerate(shortcut['components']):
                        if component['type'] == "text input":
                            inputs[component['label']] = st.session_state.get(f'text_input_{idx}_{i}', "")
                        elif component['type'] == "selector":
                            inputs[component['label']] = st.session_state.get(f'selector_{idx}_{i}', "")
                        elif component['type'] == "multi selector":
                            inputs[component['label']] = st.session_state.get(f'multi_selector_{idx}_{i}', [])
                
                    prompt_template = st.session_state[f'prompt_template_{idx}'].replace("{", "{{").replace("}", "}}")
                    for key in inputs.keys():
                        prompt_template = prompt_template.replace(f"{{{{{key}}}}}", f"{inputs[key]}")
                
                    try:
                        prompt = prompt_template.replace("{{", "{").replace("}}", "}")
                        prompt_with_line_breaks = prompt.replace("\n", "<br>")
                        st.markdown(prompt_with_line_breaks.replace('\n','\n'), unsafe_allow_html=True)
                    except KeyError as e:
                        st.error(f"缺少必需的輸入: {e}")
                st.write("\n")
                colA,colB = st.columns(2)
                with colB:
                    if shortcut['components'] and st.session_state[f'prompt_template_{idx}'].strip():
                        if len(st.session_state.get('exported_shortcuts', [])) < 4 and shortcut['name'] not in [s['name'] for s in st.session_state.get('exported_shortcuts', [])]:
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("輸出到對話頁面", key=f'export_to_chat_{idx}'):
                                    if 'exported_shortcuts' not in st.session_state:
                                        st.session_state['exported_shortcuts'] = []
                                    shortcut['target'] = 'chat'
                                    st.session_state['exported_shortcuts'].append(shortcut)
                                    save_shortcuts()
                                    st.success("成功輸出，請至對話頁查看")
                                    time.sleep(1)
                                    st.session_state['exported_shortcuts'].append(shortcut['name'])
                                    st.rerun()
                            with col2:
                                if st.button("輸出到AI生圖頁", key=f'export_to_image_{idx}'):
                                    if 'exported_shortcuts' not in st.session_state:
                                        st.session_state['exported_shortcuts'] = []
                                    shortcut['target'] = 'image'
                                    st.session_state['exported_shortcuts'].append(shortcut)
                                    save_shortcuts()
                                    st.success("成功輸出，請至AI生圖頁查看")
                                    time.sleep(1)
                                    st.session_state['exported_shortcuts'].append(shortcut['name'])
                                    st.rerun()

                if len(st.session_state['shortcuts']) > 0:
                    tab_name = shortcut['name']
                    with colA:
                        if st.session_state.get('delete_confirmation') == idx:
                            confirm_col, cancel_col = st.columns(2)
                            with confirm_col:
                                st.button("確認", key=f'confirm_delete_{idx}', on_click=confirm_delete_shortcut, args=(idx, True))
                            with cancel_col:
                                st.button("取消", key=f'cancel_delete_{idx}', on_click=confirm_delete_shortcut, args=(None, False))
                        else:
                            if st.button(f"刪除 {tab_name}", key=f'delete_tab_{idx}', on_click=lambda: confirm_delete_shortcut(idx)):
                                st.session_state['delete_confirmation'] = idx

#%% 頭像頁面
def select_avatar(name, image):
    if st.session_state['user_avatar'] != image:
        st.session_state['user_avatar'] = image
        settings['user_avatar'] = image
        save_settings(settings)
        st.session_state['avatar_updated'] = True

def display_avatars():
    cols = st.columns(6)
    for i, (name, image) in enumerate(avatars.items()):
        with cols[i % 6]:
            st.image(f"data:image/png;base64,{image}", use_column_width=True)
            st.button("選擇", key=name, on_click=select_avatar, args=(name, image))

if selected == "頭像":
    st.write("\n")
    st.markdown(f"""
        <div style='text-align: center;'>
            <div style='display: inline-block; border-radius: 60%; overflow: hidden; border: 0px; background: linear-gradient(-135deg, #35FAF9 0%, rgba(124, 45, 231, 0.8) 100%);'>
                <img src="data:image/png;base64,{st.session_state['user_avatar']}" style='width: 150px; border-radius: 50%;'/>
            </div>
            <p>\n</p>
        </div>
    """, unsafe_allow_html=True)
    st.write("\n")
    st.write("\n")
    display_avatars()

    settings['user_avatar'] = st.session_state['user_avatar']
    save_settings(settings)
