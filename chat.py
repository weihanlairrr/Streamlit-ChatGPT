import streamlit as st
import base64
from streamlit_option_menu import option_menu
import requests
from openai import AsyncOpenAI
import asyncio
import json
import time
import markdown2  # 取代 markdown

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
def save_chat_history(chat_history):
    with open('chat_history.json', 'w') as f:
        json.dump(chat_history, f)

def load_chat_history():
    try:
        with open('chat_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# 保存和載入 shortcuts 的函數
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

# 初始化設置
settings = load_settings()
chat_history = load_chat_history()
load_shortcuts()

st.markdown("""
    <style>
    .stButton > button {
        padding: 5px 20px;
        background: linear-gradient(135deg, rgba(0, 192, 251, 0.7) 30%, rgba(3, 93, 229, 0.7) 100%);
        color: #2B2727;
        border: none;
        border-radius: 5px;
        font-size: 18px;
        cursor: pointer;
        margin: 5px 0;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(83, 138, 217, 0.8) 0%, rgba(124, 45, 231, 0.8) 100%);
    }
    .stRadio > div {
        display: flex; justify-content: center; padding: 5px 20px; border: none; border-radius: 5px; background: linear-gradient(135deg, rgba(0, 192, 251, 0.7) 30%, rgba(3, 93, 229, 0.7) 100%);
    }
    p {
        margin: 0;  /* 移除 p 元素的預設邊距 */
    }
    .message {
        white-space: pre-wrap;
    }
    .custom-success {
        background-color: #d4edda;
        color: black;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #c3e6cb;
    }
    .custom-warning {
        background-color: #f8d7da;
        color: black;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)



def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 在代碼的開頭或其他初始化變量的位置
if 'prompt_submitted' not in st.session_state:
    st.session_state['prompt_submitted'] = False

# 在頁面載入時重置 prompt_submitted 標記
st.session_state['prompt_submitted'] = False

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
for key, default_value in [
    ('chatbot_api_key', settings.get('chatbot_api_key', '')),
    ('replicate_api_key', settings.get('replicate_api_key', '')),
    ('perplexity_api_key', settings.get('perplexity_api_key', '')),
    ('open_ai_model', settings.get('open_ai_model', 'gpt-4o')),
    ('perplexity_model', settings.get('perplexity_model', 'llama-3-sonar-large-32k-chat')),
    ('perplexity_temperature', settings.get('perplexity_temperature', 0.5)),
    ('perplexity_top_p', settings.get('perplexity_top_p', 0.5)),
    ('perplexity_presence_penalty', settings.get('perplexity_presence_penalty', 0.0)),
    ('perplexity_length_penalty', settings.get('perplexity_length_penalty', 0.0)),
    ('perplexity_max_tokens', settings.get('perplexity_max_tokens', 1000)),
    ('perplexity_system_prompt', settings.get('perplexity_system_prompt', '')),
    ('gpt_system_prompt', settings.get('gpt_system_prompt', '')),
    ('language', settings.get('language', '')),
    ('temperature', settings.get('temperature', 0.5)),
    ('top_p', settings.get('top_p', 0.5)),
    ('presence_penalty', settings.get('presence_penalty', 0.0)),
    ('frequency_penalty', settings.get('frequency_penalty', 0.0)),
    ('max_tokens', settings.get('max_tokens', 1000)),
    ('content', ''),
    ('reset_confirmation', False),
    ('tabs', ["對話 1"]),
    ('current_tab', 1),
    ('chat_started', False),
    ('api_key_removed', False),
    ('model_type', 'ChatGPT'),
    ('user_avatar_chatgpt', settings.get('user_avatar_chatgpt', user_avatar_default)),
    ('user_avatar_perplexity', settings.get('user_avatar_perplexity', user_avatar_default)),
    ('prompt_submitted', False)  # 初始化 prompt_submitted 鍵
]:
    if key not in st.session_state:
        st.session_state[key] = default_value


if f"messages_ChatGPT_{st.session_state['current_tab']}" not in st.session_state:
    st.session_state[f"messages_ChatGPT_{st.session_state['current_tab']}"] = chat_history.get('ChatGPT_1', [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if not st.session_state['chatbot_api_key'] else "請問需要什麼協助？"}])

if f"messages_Perplexity_{st.session_state['current_tab']}" not in st.session_state:
    st.session_state[f"messages_Perplexity_{st.session_state['current_tab']}"] = chat_history.get('Perplexity_1', [{"role": "assistant", "content": "請輸入您的 Perplexity API Key" if not st.session_state['perplexity_api_key'] else "請問需要什麼協助？"}])

if 'tab_name_1' not in st.session_state:
    st.session_state['tab_name_1'] = "對話 1"

# 根據模型選擇設置 user_avatar
if st.session_state['model_type'] == "ChatGPT":
    st.session_state['user_avatar'] = st.session_state['user_avatar_chatgpt']
else:
    st.session_state['user_avatar'] = st.session_state['user_avatar_perplexity']
    
def select_avatar(name, image):
    # 檢查頭像是否真的改變
    if st.session_state['model_type'] == "ChatGPT":
        if st.session_state['user_avatar_chatgpt'] != image:
            st.session_state['user_avatar_chatgpt'] = image
            st.session_state['user_avatar'] = image  # 更新当前使用的头像
            settings['user_avatar_chatgpt'] = image
            save_settings(settings)
            # 設定標記，表示頭像已更新
            st.session_state['avatar_updated'] = True
    else:
        if st.session_state['user_avatar_perplexity'] != image:
            st.session_state['user_avatar_perplexity'] = image
            st.session_state['user_avatar'] = image  # 更新当前使用的头像
            settings['user_avatar_perplexity'] = image
            save_settings(settings)
            # 設定標記，表示頭像已更新
            st.session_state['avatar_updated'] = True


def display_avatars():
    cols = st.columns(6)
    for i, (name, image) in enumerate(avatars.items()):
        with cols[i % 6]:
            st.image(f"data:image/png;base64,{image}", use_column_width=True)
            st.button("選擇", key=name, on_click=select_avatar, args=(name, image))

# 初始化狀態變量
if 'avatar_selected' not in st.session_state:
    st.session_state['avatar_selected'] = False

async def get_openai_response(client, model, messages, temperature, top_p, presence_penalty, frequency_penalty, max_tokens, system_prompt):
    try:
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        if st.session_state['language']:
            prompt = messages[-1]['content'] + f" 請使用{st.session_state['language']}回答。你的回答不需要提到你會使用{st.session_state['language']}。"
            messages[-1]['content'] = prompt
        else:
            prompt = messages[-1]['content'] + " 請使用繁體中文回答。你的回答不需要提到你會使用繁體中文。"
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
                yield streamed_text
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

def generate_perplexity_response(prompt, history, model, temperature, top_p, presence_penalty, max_tokens, system_prompt):
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {st.session_state['perplexity_api_key']}"
        }

        # 添加語言設定到 prompt
        if st.session_state['language']:
            prompt = prompt + f" 請使用{st.session_state['language']}回答。你的回答不需要提到你會使用{st.session_state['language']}。"
        else:
            prompt = prompt + " 請使用繁體中文回答。你的回答不需要提到你會使用繁體中文。"

        # 構建歷史上下文
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

        # 處理流式回應
        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    json_data = json.loads(decoded_line[len("data: "):])
                    if "choices" in json_data and len(json_data["choices"]) > 0:
                        chunk = json_data["choices"][0]["delta"].get("content", "")
                        full_response += chunk
                        yield full_response
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
    
def confirm_reset_chat():
    confirm, cancel = st.columns(2)
    with confirm:
        if st.button("確認", key="confirm_reset"):
            reset_chat()
            st.rerun()
    with cancel:
        if st.button("取消", key="cancel_reset", on_click=cancel_reset_chat):
            pass

def reset_chat():
    key = f"messages_{st.session_state['model_type']}_{st.session_state['current_tab']}"
    if st.session_state['model_type'] == 'ChatGPT':
        st.session_state[key] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if not st.session_state['chatbot_api_key'] else "請問需要什麼協助？"}]
    else:
        st.session_state[key] = [{"role": "assistant", "content": "請輸入您的 Perplexity API Key" if not st.session_state['perplexity_api_key'] else "請問需要什麼協助？"}]
    st.session_state['reset_confirmation'] = False
    st.session_state['chat_started'] = False
    st.session_state['api_key_removed'] = False

    # 保存重置後的聊天歷史
    chat_history[st.session_state['model_type'] + '_' + str(st.session_state['current_tab'])] = st.session_state[key]
    save_chat_history(chat_history)

def update_model_params():
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

def format_message(text):
    html_content = markdown2.markdown(text)
    html_content = html_content.replace("&nbsp;", " ")
    return html_content


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

def message_func(text, is_user=False, is_df=False):
    model_url = f"data:image/png;base64,{assistant_avatar}"
    user_url = f"data:image/png;base64,{st.session_state['user_avatar']}"

    avatar_url = model_url
    if is_user:
        avatar_url = user_url
        message_alignment = "flex-end"
        message_bg_color = "linear-gradient(135deg, #00C0FB 0%, #035DE5 100%)"
        avatar_class = "user-avatar"
        avatar_size = "width: 30px; height: 30;"
        st.markdown(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: {message_alignment};">
                    <div class="message-container" style="background: {message_bg_color}; color: white; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 10px; font-size: 15px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                        {format_message(text)} \n </div>
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

        if is_df:
            st.markdown(
                f"""
                    <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: {message_alignment};">
                        <img src="{model_url}" class="{avatar_class}" alt="avatar" style="{avatar_size}" />
                    </div>
                    """,
                unsafe_allow_html=True,
            )
            st.write(text)
            return
        else:
            text = format_message(text)

        st.markdown(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: {message_alignment};">
                    <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="{avatar_size}" />
                    <div class="message-container" style="background: {message_bg_color}; color: #2B2727; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 5px; margin-left: 5px; font-size: 15px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                        {text} \n </div>
                </div>
                """,
            unsafe_allow_html=True,
        )

async def handle_prompt_submission(prompt, current_tab_key):
    if st.session_state['model_type'] == "ChatGPT":
        client = AsyncOpenAI(api_key=st.session_state['chatbot_api_key'])
        message_func(prompt, is_user=True)

        thinking_placeholder = st.empty()
        st.session_state[current_tab_key].append({"role": "assistant", "content": "Thinking..."})
        with thinking_placeholder.container():
            message_func("Thinking...", is_user=False)

        response_container = st.empty()
        messages = st.session_state[current_tab_key] + [{"role": "user", "content": prompt}]
        full_response = ""
        async for response_message in get_openai_response(client, st.session_state['open_ai_model'], messages, st.session_state['temperature'], st.session_state['top_p'], st.session_state['presence_penalty'], st.session_state['frequency_penalty'], st.session_state['max_tokens'], st.session_state['gpt_system_prompt']):
            if "Thinking..." in [msg['content'] for msg in st.session_state[current_tab_key] if msg['role'] == 'assistant']:
                st.session_state[current_tab_key] = [msg for msg in st.session_state[current_tab_key] if msg['content'] != "Thinking..."]
                thinking_placeholder.empty()
            full_response = response_message
            response_container.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: flex-start;">
                    <img src="data:image/png;base64,{assistant_avatar_gpt}" class="bot-avatar" alt="avatar" style="width: 45px; height: 28px;" />
                    <div class="message-container" style="background: #F1F1F1; color: 2B2727; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 5px; margin-left: 5px; font-size: 15px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                        {format_message(full_response)} \n </div>
                </div>
            """, unsafe_allow_html=True)
        st.session_state[current_tab_key].append({"role": "assistant", "content": full_response})
        response_container.empty()
        message_func(full_response, is_user=False)
        chat_history[st.session_state['model_type'] + '_' + str(st.session_state['current_tab'])] = st.session_state[current_tab_key]
        save_chat_history(chat_history)

    elif st.session_state['model_type'] == "Perplexity":
        message_func(prompt, is_user=True)

        thinking_placeholder = st.empty()
        st.session_state[current_tab_key].append({"role": "assistant", "content": "Thinking..."})
        with thinking_placeholder.container():
            message_func("Thinking...", is_user=False)

        response_container = st.empty()
        full_response = ""

        history = st.session_state[current_tab_key]
        for response_message in generate_perplexity_response(
                prompt,
                history,
                st.session_state['perplexity_model'],
                st.session_state['temperature'],
                st.session_state['top_p'],
                st.session_state['presence_penalty'],
                st.session_state['max_tokens'],
                st.session_state['perplexity_system_prompt']):

            if "Thinking..." in [msg['content'] for msg in st.session_state[current_tab_key] if msg['role'] == 'assistant']:
                st.session_state[current_tab_key] = [msg for msg in st.session_state[current_tab_key] if msg['content'] != "Thinking..."]
                thinking_placeholder.empty()

            full_response = response_message
            response_container.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: flex-start;">
                    <img src="data:image/png;base64,{assistant_avatar_perplexity}" class="bot-avatar" alt="avatar" style="width: 45px; height: 28px;" />
                    <div class="message-container" style="background: #F1F1F1; color: #2B2727; border-radius: 15px; padding: 10px 15px 10px 15px; margin-right: 5px; margin-left: 5px; font-size: 15px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                        {format_message(full_response)} \n </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.session_state[current_tab_key].append({"role": "assistant", "content": full_response})
        response_container.empty()
        message_func(full_response, is_user=False)
        chat_history[st.session_state['model_type'] + '_' + str(st.session_state['current_tab'])] = st.session_state[current_tab_key]
        save_chat_history(chat_history)

def update_exported_shortcuts():
    for exported_shortcut in st.session_state.get('exported_shortcuts', []):
        for shortcut in st.session_state['shortcuts']:
            if exported_shortcut['name'] == shortcut['name']:
                exported_shortcut.update(shortcut)
                
if 'expander_state' not in st.session_state:
    st.session_state['expander_state'] = True
    
def hide_expander():
    st.session_state['expander_state'] = False
    st.session_state['active_shortcut'] = None

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
            "nav-link-selected": {"background": "linear-gradient(135deg, rgba(83, 138, 217, 0.8) 0%, rgba(124, 45, 231, 0.8) 100%)", "color": "#F1f1f1"},
        }
    )
    model_toggle = st.radio("", ["ChatGPT", "Perplexity"], key="model_type", horizontal=True, label_visibility="collapsed")
    st.write("\n")
    # 根據模型選擇設置 avatar
    if model_toggle == "Perplexity":
        assistant_avatar = assistant_avatar_perplexity
        perplexity_api_key_input = st.text_input("請輸入 Perplexity API Key", value=st.session_state.get('perplexity_api_key', ''), type="password")
        if perplexity_api_key_input != st.session_state['perplexity_api_key']:
            st.session_state['perplexity_api_key'] = perplexity_api_key_input
            settings['perplexity_api_key'] = perplexity_api_key_input
            save_settings(settings)
            if not st.session_state['chat_started']:
                if perplexity_api_key_input:
                    st.session_state[f"messages_Perplexity_{st.session_state['current_tab']}"][0]['content'] = "請問需要什麼協助？"
                else:
                    st.session_state[f"messages_Perplexity_{st.session_state['current_tab']}"][0]['content'] = "請輸入您的 Perplexity API Key"
            st.rerun()

    else:
        assistant_avatar = assistant_avatar_gpt
        api_key_input = st.text_input("請輸入 OpenAI API Key", value=st.session_state.get('chatbot_api_key', ''), type="password")
        if api_key_input != st.session_state['chatbot_api_key']:
            st.session_state['chatbot_api_key'] = api_key_input
            settings['chatbot_api_key'] = api_key_input
            save_settings(settings)
            if not st.session_state['chat_started']:
                st.session_state[f"messages_ChatGPT_{st.session_state['current_tab']}"][0]['content'] = "請問需要什麼協助？" if api_key_input else "請輸入您的 OpenAI API Key"
            st.rerun()

    current_tab_key = f"messages_{st.session_state['model_type']}_{st.session_state['current_tab']}"
    if current_tab_key not in st.session_state:
        st.session_state[current_tab_key] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if st.session_state['model_type'] == "ChatGPT" and not st.session_state['chatbot_api_key'] else "請輸入您的 Perplexity API Key" if st.session_state['model_type'] == "Perplexity" and not st.session_state['perplexity_api_key'] else "請問需要什麼協助？"}]

    if selected == "對話" and 'exported_shortcuts' in st.session_state:
        api_key_entered = (st.session_state['model_type'] == "ChatGPT" and st.session_state['chatbot_api_key']) or \
                  (st.session_state['model_type'] == "Perplexity" and st.session_state['perplexity_api_key'])

        if api_key_entered and 'exported_shortcuts' in st.session_state:
            with st.expander('你的提示詞'):
                for idx, shortcut in enumerate(st.session_state['exported_shortcuts']):
                    if st.button(shortcut['name'], key=f'exported_shortcut_{idx}'):
                        st.session_state['active_shortcut'] = shortcut

if selected == "對話":
    current_tab_key = f"messages_{st.session_state['model_type']}_{st.session_state['current_tab']}"
    if current_tab_key not in st.session_state:
        st.session_state[current_tab_key] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if st.session_state['model_type'] == "ChatGPT" and not st.session_state['chatbot_api_key'] else "請輸入您的 Perplexity API Key" if st.session_state['model_type'] == "Perplexity" and not st.session_state['perplexity_api_key'] else "請問需要什麼協助？"}]

    # 顯示對話
    for msg in st.session_state[current_tab_key]:
        message_func(msg["content"], is_user=(msg["role"] == "user"))

    if st.session_state['model_type'] == "ChatGPT" and st.session_state['chatbot_api_key']:
        prompt = st.chat_input()
        if prompt:
            st.session_state['chat_started'] = True
            client = AsyncOpenAI(api_key=st.session_state['chatbot_api_key'])
            st.session_state[current_tab_key].append({"role": "user", "content": prompt})
            message_func(prompt, is_user=True)
            
            # 顯示 "Thinking..." 訊息
            thinking_placeholder = st.empty()
            st.session_state[current_tab_key].append({"role": "assistant", "content": "Thinking..."})
            with thinking_placeholder.container():
                message_func("Thinking...", is_user=False)
            
            response_container = st.empty()
            messages = st.session_state[current_tab_key] + [{"role": "user", "content": prompt}]
            full_response = ""
                
            async def stream_openai_response():
                async for response_message in get_openai_response(client, st.session_state['open_ai_model'], messages, st.session_state['temperature'], st.session_state['top_p'], st.session_state['presence_penalty'], st.session_state['frequency_penalty'], st.session_state['max_tokens'], st.session_state['gpt_system_prompt']):
                    # 清除 "Thinking..." 訊息並開始流式回覆
                    if "Thinking..." in [msg['content'] for msg in st.session_state[current_tab_key] if msg['role'] == 'assistant']:
                        st.session_state[current_tab_key] = [msg for msg in st.session_state[current_tab_key] if msg['content'] != "Thinking..."]
                        thinking_placeholder.empty()
                    
                    full_response = response_message
                    response_container.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: flex-start;">
                            <img src="data:image/png;base64,{assistant_avatar_gpt}" class="bot-avatar" alt="avatar" style="width: 45px; height: 28px;" />
                            <div class="message-container" style="background: #F1F1F1; color: 2B2727; border-radius: 15px; padding: 10px; margin-right: 5px; margin-left: 5px; font-size: 15px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                                {format_message(full_response)} \n </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.session_state[current_tab_key].append({"role": "assistant", "content": full_response})
                response_container.empty()
                message_func(full_response, is_user=False)
                chat_history[st.session_state['model_type'] + '_' + str(st.session_state['current_tab'])] = st.session_state[current_tab_key]
                save_chat_history(chat_history)
    
            asyncio.run(stream_openai_response())
    
    if st.session_state['model_type'] == "Perplexity" and st.session_state['perplexity_api_key']:
        prompt = st.chat_input()
        if prompt:
            st.session_state['chat_started'] = True
            current_tab_key = f"messages_{st.session_state['model_type']}_{st.session_state['current_tab']}"
            st.session_state[current_tab_key].append({"role": "user", "content": prompt})
            message_func(prompt, is_user=True)
    
            # 顯示 "Thinking..." 訊息
            thinking_placeholder = st.empty()
            st.session_state[current_tab_key].append({"role": "assistant", "content": "Thinking..."})
            with thinking_placeholder.container():
                message_func("Thinking...", is_user=False)
    
            response_container = st.empty()
            full_response = ""
    
            # 構建歷史對話
            history = st.session_state[current_tab_key]
    
            # 處理流式回應並逐步更新界面
            for response_message in generate_perplexity_response(
                    prompt,
                    history,
                    st.session_state['perplexity_model'],
                    st.session_state['temperature'],
                    st.session_state['top_p'],
                    st.session_state['presence_penalty'],
                    st.session_state['max_tokens'],
                    st.session_state['perplexity_system_prompt']):
    
                # 清除 "Thinking..." 訊息並開始流式回覆
                if "Thinking..." in [msg['content'] for msg in st.session_state[current_tab_key] if msg['role'] == 'assistant']:
                    st.session_state[current_tab_key] = [msg for msg in st.session_state[current_tab_key] if msg['content'] != "Thinking..."]
                    thinking_placeholder.empty()
    
                full_response = response_message
                response_container.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: flex-start;">
                        <img src="data:image/png;base64,{assistant_avatar_perplexity}" class="bot-avatar" alt="avatar" style="width: 45px; height: 28px;" />
                        <div class="message-container" style="background: #F1F1F1; color: #2B2727; border-radius: 15px; padding: 10px; margin-right: 5px; margin-left: 5px; font-size: 15px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                            {format_message(full_response)} \n </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
            # 最後將完整的回應添加到會話狀態中
            st.session_state[current_tab_key].append({"role": "assistant", "content": full_response})
            response_container.empty()
            message_func(full_response, is_user=False)
            chat_history[st.session_state['model_type'] + '_' + str(st.session_state['current_tab'])] = st.session_state[current_tab_key]
            save_chat_history(chat_history)

    with st.sidebar:
        sidebar_placeholder = st.empty()
        sidebar_placeholder.empty()
        st.sidebar.divider()
        st.button("重置對話", on_click=lambda: st.session_state.update({'reset_confirmation': True}), use_container_width=True)
        if st.session_state.get('reset_confirmation', False):
            confirm_reset_chat()

    # 當使用者點擊送出按鈕後的邏輯處理
    if 'active_shortcut' in st.session_state and st.session_state.get('active_shortcut') is not None:
        shortcut = st.session_state['active_shortcut']
        inputs = {} 
        expander_placeholder = st.empty()
        with expander_placeholder.expander(f'{shortcut["name"]}', expanded=True):
            for i, component in enumerate(shortcut['components']):
                if component['type'] == "text input":
                    inputs[component['label']] = st.text_input(component['label'], key=f'shortcut_text_input_{i}')
                elif component['type'] == "selector":
                    inputs[component['label']] = st.selectbox(component['label'], component['options'], key=f'shortcut_selector_{i}')
                elif component['type'] == "multi selector":
                    inputs[component['label']] = st.multiselect(component['label'], component['options'], key=f'shortcut_multi_selector_{i}')
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("隱藏", on_click=hide_expander):
                    st.session_state['active_shortcut'] = None
                    expander_placeholder.empty()
                        
            with col2:
                提示詞模板 = st.button("送出")
                    
        if 提示詞模板 and not st.session_state['prompt_submitted']:
            st.session_state['active_shortcut'] = None  # 立刻停止顯示 prompt template expander
            st.session_state['expander_state'] = False
            expander_placeholder.empty()
            prompt_template = shortcut['prompt_template'].replace("{", "{{").replace("}", "}}")
            for key in inputs.keys():
                prompt_template = prompt_template.replace(f"{{{{{key}}}}}", f"{inputs[key]}")
            try:
                prompt = prompt_template.replace("{{", "{").replace("}}", "}")
                st.session_state[current_tab_key].append({"role": "user", "content": prompt})
        
                # 使用 asyncio.run 來運行異步函數
                asyncio.run(handle_prompt_submission(prompt, current_tab_key))
        
                # 將提示提交標記設為 True
                st.session_state['prompt_submitted'] = True
            except KeyError as e:
                st.error(f"缺少必需的輸入: {e}")

    # 在頁面載入時重置 prompt_submitted 標記
    if 'prompt_submitted' in st.session_state:
        del st.session_state['prompt_submitted']

if selected == "模型設定":
    col1, col2, col3 = st.columns([2, 2, 1.5])
    if st.session_state['model_type'] == "ChatGPT":
        with col1:
            st.session_state['open_ai_model'] = st.selectbox("選擇 ChatGPT 模型", ["gpt-3.5-turbo", "gpt-4o"],
                                                             index=["gpt-3.5-turbo", "gpt-4o"].index(
                                                                 st.session_state.get('open_ai_model', 'gpt-4o')),
                                                             help="4：每百萬tokens = 20美元；3.5-turbo價格為其1/10")
        with col2:
            st.text_input("指定使用的語言", key='language_input', value=st.session_state.get('language', ''), help="預設使用繁體中文。如要英文，請直接用中文輸入「英文」。", on_change=update_language)
        with col3:
            st.number_input("Tokens 上限", key='max_tokens_input', min_value=0, value=st.session_state.get('max_tokens', 1000), help="要生成的最大標記數量。", on_change=update_max_tokens)
        st.write("\n")
        st.text_area("角色設定", value=st.session_state.get('gpt_system_prompt', ''),
                     placeholder="你是一個友好且資深的英文老師。你的目標是幫助使用者提高他們的語言能力，並且用簡單易懂的方式解釋概念。你應該耐心回答問題，並鼓勵學生提出更多問題。",
                     help="用於給模型提供初始指導。", key="gpt_system_prompt_input", on_change=update_gpt_system_prompt, height=300)
        st.write("\n")
        with st.expander("模型參數", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.slider("選擇 Temperature", 
                          min_value=0.0, max_value=2.0, step=0.1, 
                          value=st.session_state['temperature'], 
                          help="較高的值會使輸出更隨機，而較低的值則會使其更加集中和確定性。一般建議只更改此參數或 Top P 中的一個，而不要同時更改。",
                          on_change=update_and_save_setting, args=('temperature',), kwargs={'value': st.session_state['temperature']})
                st.slider("選擇 Presence Penalty", 
                          min_value=-2.0, max_value=2.0, step=0.1, 
                          value=st.session_state['presence_penalty'], 
                          help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。",
                          on_change=update_and_save_setting, args=('presence_penalty',), kwargs={'value': st.session_state['presence_penalty']})
            with col2:
                st.slider("選擇 Top P", 
                          min_value=0.0, max_value=1.0, step=0.1, 
                          value=st.session_state['top_p'], 
                          help="基於核心機率的採樣，模型會考慮概率最高的top_p個標記的預測結果。當該參數為0.1時，代表只有包括前10%概率質量的標記將被考慮。一般建議只更改這個參數或 Temperature 中的一個，而不要同時更改。",
                          on_change=update_and_save_setting, args=('top_p',), kwargs={'value': st.session_state['top_p']})
                st.slider("選擇 Frequency Penalty", 
                          min_value=-2.0, max_value=2.0, step=0.1, 
                          value=st.session_state['frequency_penalty'], 
                          help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。",
                          on_change=update_and_save_setting, args=('frequency_penalty',), kwargs={'value': st.session_state['frequency_penalty']})
    elif st.session_state['model_type'] == "Perplexity":
        with col1:
            # 定義模型名稱的映射
            perplexity_model_options = {
                "sonar-large-32k-chat": "llama-3-sonar-large-32k-chat",
                "sonar-large-32k-online": "llama-3-sonar-large-32k-online",
                "llama-3-70b-instruct": "llama-3-70b-instruct",
                "llama-3-8b-instruct": "llama-3-8b-instruct"        
            }
            # 顯示簡化後的模型名稱
            reverse_mapping = {v: k for k, v in perplexity_model_options.items()}
            selected_model_key = reverse_mapping.get(st.session_state['perplexity_model'], "sonar-large-32k-chat")
            selected_model = st.selectbox("選擇 Sonar 或 Llama3 模型", list(perplexity_model_options.keys()), index=list(perplexity_model_options.keys()).index(selected_model_key), help="70b-instruct：每百萬tokens = 2.75美元；8b-instruct：每百萬tokens = 0.25美元")
            st.session_state['perplexity_model'] = perplexity_model_options[selected_model]
        with col2:
            st.text_input("指定使用的語言", key='language_input', value=st.session_state.get('language', ''), help="預設使用繁體中文。如要英文，請直接用中文輸入「英文」。", on_change=update_language)
        with col3:
            st.number_input("Tokens 上限", key='max_tokens_input', min_value=0, value=st.session_state.get('max_tokens', 1000), help="要生成的最大標記數量。", on_change=update_max_tokens)
        st.write("\n")
        st.text_area("角色設定", value=st.session_state.get('perplexity_system_prompt', ''),placeholder="你是一個專業的科技支援工程師。你的目標是幫助用戶解決各種技術問題，無論是硬體還是軟體問題。你應該詳細解釋解決方案，並確保用戶理解每一步驟。", help="用於給模型提供初始指導。", key="perplexity_system_prompt_input", on_change=update_perplexity_system_prompt, height=300)
        st.write("\n")
        with st.expander("模型參數",expanded=True):     
            col1, col2 = st.columns(2)
            with col1:
                st.slider("選擇 Temperature", 
                          min_value=0.0, max_value=2.0, step=0.1, 
                          value=st.session_state['temperature'], 
                          help="較高的值會使輸出更隨機，而較低的值則會使其更加集中和確定性。",
                          on_change=update_and_save_setting, args=('temperature',), kwargs={'value': st.session_state['temperature']})
            with col2:
                st.slider("選擇 Top P", 
                          min_value=0.0, max_value=1.0, step=0.1, 
                          value=st.session_state['top_p'], 
                          help="基於核心機率的採樣，模型會考慮概率最高的top_p個標記的預測結果。當該參數為0.1時，代表只有包括前10%概率質量的標記將被考慮。一般建議只更改這個參數或 Temperature 中的一個，而不要同時更改。",
                          on_change=update_and_save_setting, args=('top_p',), kwargs={'value': st.session_state['top_p']})
                
            st.slider("選擇 Presence Penalty", 
                          min_value=-2.0, max_value=2.0, step=0.1, 
                          value=st.session_state['presence_penalty'], 
                          help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。",
                          on_change=update_and_save_setting, args=('presence_penalty',), kwargs={'value': st.session_state['presence_penalty']})


    # 保存模型設置
    settings['open_ai_model'] = st.session_state['open_ai_model']
    settings['perplexity_model'] = st.session_state['perplexity_model']
    settings['perplexity_temperature'] = st.session_state['temperature']
    settings['perplexity_top_p'] = st.session_state['top_p']
    settings['perplexity_presence_penalty'] = st.session_state['presence_penalty']
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
    
if selected == "提示詞":
    # 初始化 session state 變量
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
            st.experimental_rerun()
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
        if len(st.session_state['shortcuts']) > 1:
            deleted_shortcut = st.session_state['shortcuts'].pop(index)
            st.session_state['shortcut_names'].pop(index)
            st.session_state['current_shortcut'] = max(0, index - 1)
            st.session_state['delete_confirmation'] = None
    
            st.session_state['exported_shortcuts'] = [
                shortcut for shortcut in st.session_state['exported_shortcuts']
                if shortcut['name'] != deleted_shortcut['name']
            ]
    
            save_shortcuts()
            # 手動更新變數來觸發重繪
            st.session_state['update_trigger'] = not st.session_state.get('update_trigger', False)


    def cancel_delete_shortcut():
        st.session_state['delete_confirmation'] = None

    def confirm_delete_shortcut(index):
        if st.session_state.get('delete_confirmation') == index:
            confirm, cancel = st.columns(2)
            with confirm:
                if st.button("確認", key=f"confirm_delete_{index}_confirm"):
                    delete_shortcut(index)
                    st.session_state['delete_confirmation'] = None  # Reset the confirmation state after deletion
                    st.experimental_rerun()  # Rerun to reflect the deletion
            with cancel:
                if st.button("取消", key=f"cancel_delete_{index}_cancel", on_click=cancel_delete_shortcut):
                    pass
        else:
            st.session_state['delete_confirmation'] = index


    def update_prompt_template(idx):
        st.session_state['shortcuts'][idx]['prompt_template'] = st.session_state[f'prompt_template_{idx}']
        update_exported_shortcuts()
        save_shortcuts()

    def update_exported_shortcuts():
        for exported_shortcut in st.session_state.get('exported_shortcuts', []):
            for shortcut in st.session_state['shortcuts']:
                if exported_shortcut['name'] == shortcut['name']:
                    exported_shortcut.update(shortcut)

    def update_shortcut_name(idx):
        new_name = st.session_state[f'shortcut_name_{idx}']
        if new_name != st.session_state['shortcuts'][idx]['name']:
            old_name = st.session_state['shortcuts'][idx]['name']
            st.session_state['shortcuts'][idx]['name'] = new_name
    
            # 同步更新 exported_shortcuts 中的名稱
            for exported_shortcut in st.session_state['exported_shortcuts']:
                if exported_shortcut['name'] == old_name:
                    exported_shortcut['name'] = new_name
    
            save_shortcuts()
            # 手動更新變數來觸發重繪
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

                # 初始化 prompt_template 鍵
                if f'prompt_template_{idx}' not in st.session_state:
                    st.session_state[f'prompt_template_{idx}'] = shortcut['prompt_template']

                col1, col2 = st.columns([1, 2.5])
                with col1:
                    component_type = st.selectbox("選擇要新增的元件類型", ["文字輸入", "選單", "多選選單"], key=f'component_type_{idx}')
                with col2:
                    new_name = st.text_input("提示詞名稱", value=shortcut['name'], key=f'shortcut_name_{idx}', on_change=update_shortcut_name, args=(idx,))
                    if new_name.strip() == "":
                        st.markdown("<div class='custom-warning'>名稱不能為空</div>", unsafe_allow_html=True)

                if component_type == "文字輸入":
                    with st.expander("建立文字變數", expanded=True):
                        label = st.text_input("變數名稱", value=st.session_state['new_component'].get('label', ''), key=f'text_input_label_{idx}')
                        if st.button("新增 文字輸入", key=f'add_text_input_{idx}'):
                            if label:
                                shortcut['components'].append({"type": "text input", "label": label})
                                st.session_state['new_component']['label'] = ''
                                reset_new_component()
                                update_exported_shortcuts()
                                save_shortcuts()
                                st.markdown("<div class='custom-success'>已成功新增</div>", unsafe_allow_html=True)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.markdown("<div class='custom-warning'>標籤為必填項目</div>", unsafe_allow_html=True)
                                time.sleep(1)
                                st.rerun()

                elif component_type == "選單":
                    with st.expander("建立選單變數", expanded=True):
                        label = st.text_input("變數名稱", value=st.session_state['new_component'].get('label', ''), key=f'selector_label_{idx}')
                        options = st.text_area("輸入選項（每行一個）", value=st.session_state['new_component'].get('options', ''), key=f'selector_options_{idx}').split("\n")
                        if st.button("新增 選單", key=f'add_selector_{idx}'):
                            if label and options and all(option.strip() for option in options):
                                shortcut['components'].append({"type": "selector", "label": label, "options": options})
                                st.session_state['new_component']['label'] = ''
                                st.session_state['new_component']['options'] = ''
                                reset_new_component()
                                update_exported_shortcuts()
                                save_shortcuts()
                                st.markdown("<div class='custom-success'>已成功新增</div>", unsafe_allow_html=True)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.markdown("<div class='custom-warning'>標籤和選項為必填項目</div>", unsafe_allow_html=True)
                                time.sleep(1)
                                st.rerun()

                elif component_type == "多選選單":
                    with st.expander("建立多選選單變數", expanded=True):
                        label = st.text_input("變數名稱", value=st.session_state['new_component'].get('label', ''), key=f'multi_selector_label_{idx}')
                        options = st.text_area("輸入選項（每行一個）", value=st.session_state['new_component'].get('options', ''), key=f'multi_selector_options_{idx}').split("\n")
                        if st.button("新增 多選選單", key=f'add_multi_selector_{idx}'):
                            if label and options and all(option.strip() for option in options):
                                shortcut['components'].append({"type": "multi selector", "label": label, "options": options})
                                st.session_state['new_component']['label'] = ''
                                st.session_state['new_component']['options'] = ''
                                reset_new_component()
                                update_exported_shortcuts()
                                save_shortcuts()
                                st.markdown("<div class='custom-success'>已成功新增</div>", unsafe_allow_html=True)
                                time.sleep(1)
                                st.rerun()
                                
                st.divider()
                st.subheader("你的元件組合")
                st.write("\n")
                cols = st.columns(3)
                for i, component in enumerate(shortcut['components']):
                    col = cols[i % 3]
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
                    st.text_area("", value=st.session_state[f'prompt_template_{idx}'], height=350, placeholder="用{ }代表標籤變數", key=f'prompt_template_{idx}', label_visibility="collapsed", on_change=update_prompt_template, args=(idx,))
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
                        st.markdown(prompt.replace('\n', '  \n'))
                    except KeyError as e:
                        st.error(f"缺少必需的輸入: {e}")

                if shortcut['components'] and st.session_state[f'prompt_template_{idx}'].strip():
                    if len(st.session_state.get('exported_shortcuts', [])) < 4 and shortcut['name'] not in [s['name'] for s in st.session_state.get('exported_shortcuts', [])]:
                        if st.button("輸出到對話頁面", key=f'export_to_chat_{idx}'):
                            if 'exported_shortcuts' not in st.session_state:
                                st.session_state['exported_shortcuts'] = []
                            st.session_state['exported_shortcuts'].append(shortcut)
                            save_shortcuts()
                            st.markdown("<div class='custom-success'>成功輸出，請至對話頁查看</div>", unsafe_allow_html=True)
                            time.sleep(1)
                            st.session_state['exported_shortcuts'].append(shortcut['name'])  # 防止按鈕再次出現
                            st.rerun()


                # 在tab內新增刪除按鈕，確保刪除當前tab
                st.write("\n")
                if len(st.session_state['shortcuts']) > 1:
                    if st.button("刪除提示詞", key=f'delete_tab_{idx}'):
                        confirm_delete_shortcut(idx)

                if st.session_state.get('delete_confirmation') is not None:
                    confirm_delete_shortcut(st.session_state['delete_confirmation'])

elif selected == "頭像":
    st.markdown(f"""
        <div style='text-align: center;'>
            <div style='display: inline-block; border-radius: 60%; overflow: hidden; border: 0px; background: linear-gradient(135deg, rgba(83, 138, 217, 0.8) 0%, rgba(124, 45, 231, 0.8) 100%);'>
                <img src="data:image/png;base64,{st.session_state['user_avatar']}" style='width: 150px; border-radius: 50%;'/>
            </div>
            <p>\n</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #999;'>目前的頭像</p>", unsafe_allow_html=True)
    st.write("\n")
    display_avatars()

    settings['user_avatar_chatgpt'] = st.session_state['user_avatar_chatgpt']
    settings['user_avatar_perplexity'] = st.session_state['user_avatar_perplexity']
    save_settings(settings) 
