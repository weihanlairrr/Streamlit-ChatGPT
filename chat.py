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
from openai import AsyncOpenAI, OpenAI
from PIL import Image

async def get_openai_response(client, model, messages, temperature, top_p, presence_penalty, frequency_penalty, max_tokens, system_prompt, language):
    try:
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        if st.session_state['language']:
            prompt = messages[-1]['content'] + f" 請使用{st.session_state['language']}回答。你的回答不需要提到你會使用{st.session_state['language']}。"
            messages[-1]['content'] = prompt

        response = await client.chat.completions.create(
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
        async for chunk in response:
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
            prompt = prompt + f" 請使用{st.session_state['language']}回答。你的回答不需要提到你會使用{st.session_state['language']}。"

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


# 保存和載入設置的函數
def save_settings(settings):
    with open('settings.json', 'w') as f:
        json.dump(settings, f)

def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# 保存和載入聊天歷史的函數
def save_chat_history(chat_history, model_type):
    filename = 'chat_history_gpt.json' if model_type == 'ChatGPT' else 'chat_history_perplexity.json'
    with open(filename, 'w') as f:
        json.dump(chat_history, f)

def load_chat_history(model_type):
    filename = 'chat_history_gpt.json' if model_type == 'ChatGPT' else 'chat_history_perplexity.json'
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# 保存和載入快捷方式的函數
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

# 初始化設置和聊天歷史
settings = load_settings()
chat_history_gpt = load_chat_history('ChatGPT')
chat_history_perplexity = load_chat_history('Perplexity')
load_shortcuts()

# 工具函數
def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 頭像圖片
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

# 初始化狀態變量
def init_session_state():
    if 'prompt_submitted' not in st.session_state:
        st.session_state['prompt_submitted'] = False

    for key, default_value in [
        ('chatbot_api_key', settings.get('chatbot_api_key', '')),
        ('perplexity_api_key', settings.get('perplexity_api_key', '')),
        ('open_ai_model', settings.get('open_ai_model', 'gpt-4o-mini')),
        ('perplexity_model', settings.get('perplexity_model', 'llama-3-sonar-large-32k-online')),
        ('perplexity_temperature', settings.get('perplexity_temperature', 0.5)),
        ('perplexity_top_p', settings.get('perplexity_top_p', 0.5)),
        ('perplexity_presence_penalty', settings.get('perplexity_presence_penalty', 0.0)),
        ('perplexity_max_tokens', settings.get('perplexity_max_tokens', 1000)),
        ('perplexity_system_prompt', settings.get('perplexity_system_prompt', '')),
        ('gpt_system_prompt', settings.get('gpt_system_prompt', '')),
        ('language', settings.get('language', '繁體中文')),
        ('temperature', settings.get('temperature', 0.5)),
        ('top_p', settings.get('top_p', 0.5)),
        ('presence_penalty', settings.get('presence_penalty', 0.0)),
        ('frequency_penalty', settings.get('frequency_penalty', 0.0)),
        ('max_tokens', settings.get('max_tokens', 1000)),
        ('content', ''),
        ('reset_confirmation', False),
        ('chat_started', False),
        ('api_key_removed', False),
        ('model_type', 'ChatGPT'),
        ('user_avatar_chatgpt', settings.get('user_avatar_chatgpt', user_avatar_default)),
        ('user_avatar_perplexity', settings.get('user_avatar_perplexity', user_avatar_default)),
        ('prompt_submitted', False),  
        ('reset_triggered', False)  
    ]:
        if key not in st.session_state:
            st.session_state[key] = default_value

    if 'reset_triggered' in st.session_state and st.session_state['reset_triggered']:
        if st.session_state['model_type'] == 'ChatGPT':
            st.session_state['messages_ChatGPT'] = [{"role": "assistant", "content": "請問需要什麼協助？"}]
        else:
            st.session_state['messages_Perplexity'] = [{"role": "assistant", "content": "請問需要什麼協助？"}]
    
        st.session_state['reset_triggered'] = False  
    else:
        if "messages_ChatGPT" not in st.session_state:
            st.session_state["messages_ChatGPT"] = chat_history_gpt.get('ChatGPT', [])
            if not st.session_state["messages_ChatGPT"]:
                st.session_state["messages_ChatGPT"] = [{"role": "assistant", "content": "請問需要什麼協助？"}]
    
        if "messages_Perplexity" not in st.session_state:
            st.session_state["messages_Perplexity"] = chat_history_perplexity.get('Perplexity', [])
            if not st.session_state["messages_Perplexity"]:
                st.session_state["messages_Perplexity"] = [{"role": "assistant", "content": "請問需要什麼協助？"}]
    
    if 'reset_confirmed' not in st.session_state:
        st.session_state['reset_confirmed'] = False

    if st.session_state['model_type'] == "ChatGPT":
        st.session_state['user_avatar'] = st.session_state['user_avatar_chatgpt']
    else:
        st.session_state['user_avatar'] = st.session_state['user_avatar_perplexity']

    if 'avatar_selected' not in st.session_state:
        st.session_state['avatar_selected'] = False

    if 'expander_state' not in st.session_state:
        st.session_state['expander_state'] = True

init_session_state()


# 自訂樣式
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
            background: linear-gradient(135deg, #58BBFF 30%, #3380FF 100%);
            color: #2B2727;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
            margin: 5px 0;
            width: 100%;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, rgba(0, 192, 251, 0.7) 30%, #30A2FD 100%);
        }
        .stRadio > div {
            display: flex;
            justify-content: center;
            padding: 5px 20px;
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
        </style>
        """,
        unsafe_allow_html=True
    )
              

def select_avatar(name, image):
    if st.session_state['model_type'] == "ChatGPT":
        if st.session_state['user_avatar_chatgpt'] != image:
            st.session_state['user_avatar_chatgpt'] = image
            st.session_state['user_avatar'] = image
            settings['user_avatar_chatgpt'] = image
            save_settings(settings)
            st.session_state['avatar_updated'] = True
    else:
        if st.session_state['user_avatar_perplexity'] != image:
            st.session_state['user_avatar_perplexity'] = image
            st.session_state['user_avatar'] = image
            settings['user_avatar_perplexity'] = image
            save_settings(settings)
            st.session_state['avatar_updated'] = True

def display_avatars():
    cols = st.columns(6)
    for i, (name, image) in enumerate(avatars.items()):
        with cols[i % 6]:
            st.image(f"data:image/png;base64,{image}", use_column_width=True)
            st.button("選擇", key=name, on_click=select_avatar, args=(name, image))

async def handle_prompt_submission(prompt):
    if st.session_state['model_type'] == "ChatGPT":
        client = AsyncOpenAI(api_key=st.session_state['chatbot_api_key'])
        message_func(prompt, is_user=True)

        thinking_placeholder = st.empty()
        status_text = "Thinking..."
        st.session_state["messages_ChatGPT"].append({"role": "assistant", "content": status_text})
        with thinking_placeholder.container():
            message_func(status_text, is_user=False)

        response_container = st.empty()
        full_response = ""
        
        messages = st.session_state["messages_ChatGPT"] + [{"role": "user", "content": prompt}]
        
        async for response_message in get_openai_response(client, st.session_state['open_ai_model'], messages, st.session_state['temperature'], st.session_state['top_p'], st.session_state['presence_penalty'], st.session_state['frequency_penalty'], st.session_state['max_tokens'], st.session_state['gpt_system_prompt'], st.session_state['language']):
            if status_text in [msg['content'] for msg in st.session_state["messages_ChatGPT"] if msg['role'] == 'assistant']:
                st.session_state["messages_ChatGPT"] = [msg for msg in st.session_state["messages_ChatGPT"] if msg['content'] != status_text]
                thinking_placeholder.empty()

            full_response = response_message
            response_container.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: flex-start;">
                    <img src="data:image/png;base64,{assistant_avatar_gpt}" class="bot-avatar" alt="avatar" style="width: 45px; height: 28px;" />
                    <div class="message-container" style="background: #F1F1F1; color: #2B2727; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 5px; margin-left: 5px; font-size: 16px; max-width: 80%; word-wrap: break-word; word-break: break-all;">
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

    elif st.session_state['model_type'] == "Perplexity":
        message_func(prompt, is_user=True)
        
        prev_state = st.session_state.get('prev_state', {}).get('messages_Perplexity', []).copy()
        
        thinking_placeholder = st.empty()
        status_text = "Thinking..."
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
                    <div class="message-container" style="background: #F1F1F1; color: #2B2727; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 5px; margin-left: 5px; font-size: 16px; max-width: 80%; word-wrap: break-word; word-break: break-all;">
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

        if prev_state != st.session_state["messages_Perplexity"]:
            st.session_state['prev_state'] = {'messages_Perplexity': st.session_state["messages_Perplexity"].copy()}
            st.rerun()

def update_slider(key, value):
    st.session_state[key] = value
    save_settings({
        'temperature': st.session_state['temperature'],
        'top_p': st.session_state['top_p'],
        'presence_penalty': st.session_state['presence_penalty'],
        'frequency_penalty': st.session_state['frequency_penalty']
    })

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
    
def reset_chat():
    if st.session_state['model_type'] == 'ChatGPT':
        st.session_state['chat_started'] = False
        st.session_state['messages_ChatGPT'] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if not st.session_state['chatbot_api_key'] else "請問需要什麼協助？"}]
    else:
        st.session_state['chat_started'] = False
        st.session_state['messages_Perplexity'] = [{"role": "assistant", "content": "請輸入您的 Perplexity API Key" if not st.session_state['perplexity_api_key'] else "請問需要什麼協助？"}]
    
    st.session_state['reset_confirmation'] = False
    st.session_state['api_key_removed'] = False
    st.session_state['reset_confirmed'] = True
    st.session_state['reset_triggered'] = True

    if st.session_state['model_type'] == 'ChatGPT':
        if os.path.exists('chat_history_gpt.json'):
            os.remove('chat_history_gpt.json')
    else:
        if os.path.exists('chat_history_perplexity.json'):
            os.remove('chat_history_perplexity.json')

def update_openai_api_key():
    if st.session_state['openai_api_key_input'] != st.session_state['chatbot_api_key']:
        st.session_state['chatbot_api_key'] = st.session_state['openai_api_key_input']
        settings['chatbot_api_key'] = st.session_state['chatbot_api_key']
        save_settings(settings)
        if not st.session_state['chat_started']:
            st.session_state["messages_ChatGPT"][0]['content'] = "請問需要什麼協助？"

def update_perplexity_api_key():
    if st.session_state['perplexity_api_key_input'] != st.session_state['perplexity_api_key']:
        st.session_state['perplexity_api_key'] = st.session_state['perplexity_api_key_input']
        settings['perplexity_api_key'] = st.session_state['perplexity_api_key']
        save_settings(settings)
        if not st.session_state['chat_started']:
            st.session_state["messages_Perplexity"][0]['content'] = "請問需要什麼協助？"
            
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

def update_and_save_setting(key, value):
    st.session_state[key] = value
    settings[key] = value
    save_settings(settings)

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
                 {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]},
                 {'selector': 'tbody tr:hover', 'props': [('background-color', '#ddd')]}]
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
        message_bg_color = "#F1F1F1"
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
            
def update_exported_shortcuts():
    for exported_shortcut in st.session_state.get('exported_shortcuts', []):
        for shortcut in st.session_state['shortcuts']:
            if exported_shortcut['name'] == shortcut['name']:
                exported_shortcut.update(shortcut)

def hide_expander():
    st.session_state['expander_state'] = False
    st.session_state['active_shortcut'] = None

# 側邊欄設置
with st.sidebar:
    st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" style="width: 100%; height: 100%; " />
        </div>
    """, unsafe_allow_html=True)

    selected = option_menu("",
        ["對話",'模型設定','提示詞','頭像'],
        icons=['chat-dots-fill','gear-fill','info-square-fill','person-square'], menu_icon="robot", default_index=0,
        styles={
            "container": {"padding": "0!important", "background": "linear-gradient(180deg, #e5e5e5 0%, #f5f5f5 80%)"},
            "icon": {"color": "#FF8C00", "font-size": "18px"},
            "nav-link": {"font-size": "18px", "text-align": "left", "margin":"5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background": "linear-gradient(-135deg, #6DD0FA 0%, rgba(124, 45, 231, 0.8) 100%)", "color": "#F1f1f1"},
        }
    )
    model_toggle = st.radio("", ["ChatGPT", "Perplexity"], key="model_type", horizontal=True, label_visibility="collapsed")
    st.write("\n")

    if model_toggle == "Perplexity":
        assistant_avatar = assistant_avatar_perplexity
        perplexity_api_key_input = st.text_input("請輸入 Perplexity API Key", value=st.session_state.get('perplexity_api_key', ''), type="password", key='perplexity_api_key_input', on_change=update_perplexity_api_key)
    
    else:
        assistant_avatar = assistant_avatar_gpt
        openai_api_key_input = st.text_input("請輸入 OpenAI API Key", value=st.session_state.get('chatbot_api_key', ''), type="password", key='openai_api_key_input', on_change=update_openai_api_key)


# 對話頁面
if selected == "對話" and 'exported_shortcuts' in st.session_state:
    api_key_entered = (st.session_state['model_type'] == "ChatGPT" and st.session_state['chatbot_api_key']) or \
              (st.session_state['model_type'] == "Perplexity" and st.session_state['perplexity_api_key'])

    if api_key_entered and 'exported_shortcuts' in st.session_state and not (st.session_state['model_type'] == "ChatGPT" and st.session_state['open_ai_model'] == "DALL-E"):
        with st.sidebar.expander('你的提示詞'):
            for idx, shortcut in enumerate(st.session_state['exported_shortcuts']):
                col = st.columns(1)[0]
                with col:
                    if ui.button(shortcut['name'], key=f'exported_shortcut_{idx}', style={"width": "100%", "background": "#C4DDA7", "color": "#2b2727"}):
                        st.session_state['active_shortcut'] = shortcut

    if 'exported_shortcuts' in st.session_state and not (st.session_state['model_type'] == "ChatGPT" and st.session_state['open_ai_model'] == "DALL-E"):
        with st.sidebar:
            st.divider()
            st.button("重置對話", on_click=lambda: st.session_state.update({'reset_confirmation': True}), use_container_width=True)
            if st.session_state.get('reset_confirmation', False):
                confirm_reset_chat()

    if selected == "對話":
        if st.session_state['reset_confirmed']:
            st.session_state['reset_confirmed'] = False
            
        if f"messages_{st.session_state['model_type']}" not in st.session_state:
            st.session_state[f"messages_{st.session_state['model_type']}"] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if st.session_state['model_type'] == "ChatGPT" and not st.session_state['chatbot_api_key'] else "請輸入您的 Perplexity API Key" if st.session_state['model_type'] == "Perplexity" and not st.session_state['perplexity_api_key'] else "請問需要什麼協助？"}]
        
        if not (st.session_state['model_type'] == "ChatGPT" and st.session_state['open_ai_model'] == "DALL-E"):
            for msg in st.session_state[f"messages_{st.session_state['model_type']}"]:
                message_func(msg["content"], is_user=(msg["role"] == "user"))

        if st.session_state['model_type'] == "ChatGPT":
            if not st.session_state['open_ai_model'] == "DALL-E":
                prompt = st.chat_input()
                if prompt:
                    if not st.session_state['chatbot_api_key']:
                        message_func("請輸入您的 OpenAI API Key", is_user=False)
                    else:
                        st.session_state['chat_started'] = True
                        client = AsyncOpenAI(api_key=st.session_state['chatbot_api_key'])
                        st.session_state[f"messages_{st.session_state['model_type']}"].append({"role": "user", "content": prompt})
                        message_func(prompt, is_user=True)

                        thinking_placeholder = st.empty()
                        status_text = "Thinking..."
                        st.session_state[f"messages_{st.session_state['model_type']}"].append({"role": "assistant", "content": status_text})
                        with thinking_placeholder.container():
                            message_func(status_text, is_user=False)

                        response_container = st.empty()
                        messages = st.session_state[f"messages_{st.session_state['model_type']}"] + [{"role": "user", "content": prompt}]
                        full_response = ""

                        async def stream_openai_response():
                            async for response_message in get_openai_response(client, st.session_state['open_ai_model'], messages, st.session_state['temperature'], st.session_state['top_p'], st.session_state['presence_penalty'], st.session_state['frequency_penalty'], st.session_state['max_tokens'], st.session_state['gpt_system_prompt'], st.session_state['language']):
                                if status_text in [msg['content'] for msg in st.session_state[f"messages_{st.session_state['model_type']}"] if msg['role'] == 'assistant']:
                                    st.session_state[f"messages_{st.session_state['model_type']}"] = [msg for msg in st.session_state[f"messages_{st.session_state['model_type']}"] if msg['content'] != status_text]
                                    thinking_placeholder.empty()

                                full_response = response_message
                                response_container.markdown(f"""
                                    <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: flex-start;">
                                        <img src="data:image/png;base64,{assistant_avatar_gpt}" class="bot-avatar" alt="avatar" style="width: 45px; height: 28px;" />
                                        <div class="message-container" style="background: #F1F1F1; color: #2B2727; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 5px; margin-left: 5px; font-size: 16px; max-width: 80%; word-wrap: break-word; word-break: break-all;">
                                            {format_message(full_response)} \n </div>
                                    </div>
                                """, unsafe_allow_html=True)

                            st.session_state[f"messages_{st.session_state['model_type']}"].append({"role": "assistant", "content": full_response})
                            response_container.empty()
                            message_func(full_response, is_user=False)
                            chat_history_gpt[st.session_state['model_type']] = st.session_state[f"messages_{st.session_state['model_type']}"]
                            save_chat_history(chat_history_gpt, 'ChatGPT')

                        asyncio.run(stream_openai_response())

            else:
                if st.session_state['chatbot_api_key']:
                    prompt = st.text_input("輸入提示詞")
                    negative_prompt = st.text_input("輸入不希望出現的內容（選填）")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        model_choice = st.selectbox(
                            "選擇 DALL-E 模型",
                            ("DALL-E 3", "DALL-E 2"),
                            index=0,
                            placeholder="",
                        )
                        model_choice = "dall-e-3" if model_choice == "DALL-E 3" else "dall-e-2"

                        color_preference_options = {
                            "無特定偏好": "no specific color preference",
                            "暖色調": "warm color scheme",
                            "冷色調": "cool color scheme",
                            "黑白": "black and white",
                            "柔和色調": "soft color palette",
                            "鮮豔色調": "vibrant color palette",
                            "低飽和色調": "low saturation color scheme"
                        }
                        selected_color_preference_zh = st.selectbox("色彩偏好", list(color_preference_options.keys()))
                        selected_color_preference_en = color_preference_options[selected_color_preference_zh]

                    with col2:
                        size_options = {
                            "1024x1024": "1024x1024",
                            "1792x1024": "1792x1024",
                            "1024x1792": "1024x1792"
                        }
                        selected_size = st.selectbox("圖片尺寸", list(size_options.keys()))

                        effect_options = {
                            "無特定偏好": "no specific effect preference",
                            "顆粒質感": "grainy texture",
                            "玻璃質感": "glass-like effect",
                            "紙質感": "paper texture",
                            "金屬質感": "metallic sheen",
                            "馬賽克效果": "mosaic pattern",
                            "浮雕效果": "embossed effect",
                            "陶瓷質感": "ceramic texture",
                            "黏土質感": "clay texture", 
                            "木頭質感": "wood texture",
                            "磚塊質感": "brick texture"
                        }
                        selected_effect = st.selectbox("圖片效果", list(effect_options.keys()))
                        selected_effect_en = effect_options[selected_effect]

                    with col3:
                        style_options = {
                            "無特定偏好": "no specific style preference",
                            "寫實風格": "realistic style",
                            "卡通風格": "cartoon style",
                            "水彩畫風格": "watercolor style",
                            "油畫風格": "oil painting style",
                            "素描風格": "sketch style",
                            "像素藝術": "pixel art",
                            "復古風格": "vintage style",
                            "超現實主義": "surrealism",
                            "極簡主義": "minimalism",
                            "印象派": "impressionism",
                            "抽象藝術": "abstract art",
                            "3D渲染": "3D render",
                            "普普藝術": "pop art",
                            "哥德風格": "gothic style",
                            "日式動漫": "anime style",
                            "中國水墨畫": "Chinese ink painting",
                            "拼貼藝術": "collage art",
                            "立體主義": "cubism",
                            "電影海報": "movie poster",
                            "科幻插畫": "sci-fi illustration",
                        }
                        selected_style_zh = st.selectbox("圖片風格", list(style_options.keys()))
                        selected_style_en = style_options[selected_style_zh]

                        light_options = {
                            "無特定偏好": "no specific lighting preference",
                            "攝影棚燈光": "studio lighting",
                            "自然光線": "natural lighting",
                            "舞台燈光": "stage lighting",
                            "背光效果": "backlit effect",
                            "螢光效果": "neon lighting",
                            "燭光氛圍": "candlelight ambiance"
                        }
                        selected_light_zh = st.selectbox("光線設定", list(light_options.keys()))
                        selected_light_en = light_options[selected_light_zh]

                    detail_level = st.slider("細節程度", 1, 10, 5)

                    if st.button("生成圖片"):
                        if not prompt.strip():
                            warning_placeholder = st.empty()
                            warning_placeholder.markdown("<div class='custom-warning'>請輸入提示詞</div>", unsafe_allow_html=True)
                            time.sleep(2)
                            warning_placeholder.empty()
                        else:
                            with st.spinner('圖片生成中...'):
                                if selected_effect != "無特定效果":
                                    full_prompt = f"{prompt}, with {selected_effect_en}, {selected_style_en} style, {selected_color_preference_en}, {selected_light_en}, with detail level {detail_level} out of 10"
                                else:
                                    full_prompt = f"{prompt}, {selected_style_en} style, {selected_color_preference_en}, {selected_light_en}, with detail level {detail_level} out of 10"

                                if negative_prompt:
                                    full_prompt += f". Avoid including: {negative_prompt}"

                                client = OpenAI(api_key=st.session_state['chatbot_api_key'])

                                try:
                                    response = client.images.generate(
                                        model=model_choice,
                                        prompt=full_prompt,
                                        size=selected_size,
                                        n=1
                                    )
                                    image_url = response.data[0].url
                                
                                    response = requests.get(image_url)
                                    img = Image.open(BytesIO(response.content))
                                
                                    st.image(img)
                                except Exception as e:
                                    st.error(f"圖片生成失敗：{str(e)}")

        if st.session_state['model_type'] == "Perplexity":
            prompt = st.chat_input()
            if prompt:
                if not st.session_state['perplexity_api_key']:
                    message_func("請輸入您的 Perplexity API Key", is_user=False)
                else:
                    st.session_state['chat_started'] = True
                    st.session_state[f"messages_{st.session_state['model_type']}"].append({"role": "user", "content": prompt})
                    message_func(prompt, is_user=True)

                    thinking_placeholder = st.empty()
                    status_text = "Thinking..."
                    st.session_state[f"messages_{st.session_state['model_type']}"].append({"role": "assistant", "content": status_text})
                    with thinking_placeholder.container():
                        message_func(status_text, is_user=False)

                    response_container = st.empty()
                    full_response = ""
                    history = st.session_state[f"messages_{st.session_state['model_type']}"]

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

                        if status_text in [msg['content'] for msg in st.session_state[f"messages_{st.session_state['model_type']}"] if msg['role'] == 'assistant']:
                            st.session_state[f"messages_{st.session_state['model_type']}"] = [msg for msg in st.session_state[f"messages_{st.session_state['model_type']}"] if msg['content'] != status_text]
                            thinking_placeholder.empty()

                        full_response = response_message
                        response_container.markdown(
                            f"""
                            <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: flex-start;">
                                <img src="data:image/png;base64,{assistant_avatar_perplexity}" class="bot-avatar" alt="avatar" style="width: 45px; height: 28px;" />
                                <div class="message-container" style="background: #F1F1F1; color: #2B2727; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 5px; margin-left: 5px; font-size: 16px; max-width: 80%; word-wrap: break-word; word-break: break-all;">
                                    {full_response} \n </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.session_state[f"messages_{st.session_state['model_type']}"].append({"role": "assistant", "content": full_response})
                    response_container.empty()
                    message_func(full_response, is_user=False)
                    chat_history_perplexity[st.session_state['model_type']] = st.session_state[f"messages_{st.session_state['model_type']}"]
                    save_chat_history(chat_history_perplexity, 'Perplexity')

    with st.sidebar:
        sidebar_placeholder = st.empty()
        sidebar_placeholder.empty()

    if 'active_shortcut' in st.session_state and st.session_state.get('active_shortcut') is not None:
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
            st.session_state['active_shortcut'] = None
            st.session_state['expander_state'] = False
            form_placeholder.empty()
            prompt_template = shortcut['prompt_template'].replace("{", "{{").replace("}", "}}")
            for key in inputs.keys():
                prompt_template = prompt_template.replace(f"{{{{{key}}}}}", f"{inputs[key]}")
            try:
                prompt = prompt_template.replace("{{", "{").replace("}}", "}")
                st.session_state[f"messages_{st.session_state['model_type']}"].append({"role": "user", "content": prompt})
    
                asyncio.run(handle_prompt_submission(prompt))
    
                st.session_state['prompt_submitted'] = True
            except KeyError as e:
                st.error(f"缺少必需的輸入: {e}")

    if 'prompt_submitted' in st.session_state:
        del st.session_state['prompt_submitted']


def update_open_ai_model():
    model_display_names = {"GPT-4o": "gpt-4o", "GPT-4o mini": "gpt-4o-mini", "DALL-E": "DALL-E"}
    selected_model = model_display_names[st.session_state['open_ai_model_selection']]
    st.session_state['open_ai_model'] = selected_model
    settings['open_ai_model'] = selected_model
    save_settings(settings)
    st.session_state['update_trigger'] = not st.session_state.get('update_trigger', False)

def update_perplexity_model():
    perplexity_model_display_names = {
        "Sonar-Large 32k Online": "llama-3-sonar-large-32k-online",
        "Sonar-Large 32k Chat": "llama-3-sonar-large-32k-chat",
        "Llama-3 70b Instruct": "llama-3-70b-instruct",
        "Llama-3 8b Instruct": "llama-3-8b-instruct",
        "Mixtral 8x7b Instruct": "mixtral-8x7b-instruct"
    }
    selected_model = perplexity_model_display_names[st.session_state['perplexity_model_selection']]
    st.session_state['perplexity_model'] = selected_model
    settings['perplexity_model'] = selected_model
    save_settings(settings)
    st.session_state['update_trigger'] = not st.session_state.get('update_trigger', False)

if selected == "模型設定":
    col1, col2, col3 = st.columns([2, 2, 1.5])
    if st.session_state['model_type'] == "ChatGPT":
        with col1:
            model_display_names = {"GPT-4o": "gpt-4o", "GPT-4o mini": "gpt-4o-mini", "DALL-E": "DALL-E"}
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
                "Sonar-Large 32k Online": "llama-3-sonar-large-32k-online",
                "Sonar-Large 32k Chat": "llama-3-sonar-large-32k-chat",
                "Llama-3 70b Instruct": "llama-3-70b-instruct",
                "Llama-3 8b Instruct": "llama-3-8b-instruct",
                "Mixtral 8x7b Instruct": "mixtral-8x7b-instruct"
            }
            reverse_mapping = {v: k for k, v in perplexity_model_display_names.items()}
            selected_model_key = reverse_mapping.get(st.session_state['perplexity_model'], "Sonar Large 32k Online")
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

# 提示詞頁面
if selected == "提示詞":
    if 'shortcuts' not in st.session_state:
        st.session_state['shortcuts'] = load_shortcuts()
    if 'current_shortcut' not in st.session_state:
        st.session_state['current_shortcut'] = 0
    if 'new_component' not in st.session_state:
        st.session_state['new_component'] = {"label": "", "options": ""}
    if 'shortcut_names' not in st.session_state:
        st.session_state['shortcut_names'] = [shortcut["name"] for shortcut in st.session_state['shortcuts']]
    if 'exported_shortcuts' not in st.session_state:
        st.session_state['exported_shortcuts'] = []

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
            
    def cancel_delete_shortcut():
        st.session_state['delete_confirmation'] = None

    def confirm_delete_shortcut(index=None, confirm_delete=None):
        if confirm_delete is None:
            if st.session_state.get('delete_confirmation') == index:
                confirm, cancel = st.columns(2)
                with confirm:
                    st.button("確認", key=f"confirm_delete_{st.session_state['current_shortcut']}_confirm", on_click=confirm_delete_shortcut, args=(index, True))
                with cancel:
                    st.button("取消", key=f"cancel_delete_{st.session_state['current_shortcut']}_cancel", on_click=confirm_delete_shortcut, args=(None, False))
            else:
                st.session_state['delete_confirmation'] = index
        else:
            if confirm_delete:
                delete_shortcut(index)
                st.session_state['delete_confirmation'] = None
            else:
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
                        if st.button("新增 文字輸入", key=f'add_text_input_{idx}'):
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
                        if st.button("新增 選單", key=f'add_selector_{idx}'):
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
                        if st.button("新增 多選選單", key=f'add_multi_selector_{idx}'):
                            if label and options and all(option.strip() for option in options):
                                shortcut['components'].append({"type": "multi selector", "label": label, "options": options})
                                reset_new_component()
                                update_exported_shortcuts()
                                save_shortcuts()
                                st.success("已成功新增")
                                time.sleep(1)
                                st.rerun()

                st.divider()
                st.subheader("你的元件組合")

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
                st.subheader("自訂提示詞公式")
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
                
                if shortcut['components'] and st.session_state[f'prompt_template_{idx}'].strip():
                    if len(st.session_state.get('exported_shortcuts', [])) < 4 and shortcut['name'] not in [s['name'] for s in st.session_state.get('exported_shortcuts', [])]:
                        if st.button("輸出到對話頁面", key=f'export_to_chat_{idx}'):
                            if 'exported_shortcuts' not in st.session_state:
                                st.session_state['exported_shortcuts'] = []
                            st.session_state['exported_shortcuts'].append(shortcut)
                            save_shortcuts()
                            st.success("成功輸出，請至對話頁查看")
                            time.sleep(1)
                            st.session_state['exported_shortcuts'].append(shortcut['name'])
                            st.rerun()

                if len(st.session_state['shortcuts']) > 0:
                    tab_name = shortcut['name']
                    if st.button(f"刪除 {tab_name}", key=f'delete_tab_{idx}'):
                        confirm_delete_shortcut(idx)

                if st.session_state.get('delete_confirmation') is not None:
                    confirm_delete_shortcut(st.session_state['delete_confirmation'])

# 頭像頁面
elif selected == "頭像":
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

    settings['user_avatar_chatgpt'] = st.session_state['user_avatar_chatgpt']
    settings['user_avatar_perplexity'] = st.session_state['user_avatar_perplexity']
    save_settings(settings)
