import streamlit as st
import base64
from streamlit_option_menu import option_menu
import replicate
import os
import requests
from openai import AsyncOpenAI
import asyncio
import json


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
        background: #6076DD;
    }
    .stRadio {
        display: flex;
        justify-content: center;
        padding: 5px 15px 5px 15px; /* 調整這裡的內邊距 */
        border: none;
        border-radius: 5px;
        background: linear-gradient(135deg, rgba(0, 192, 251, 0.7) 30%, rgba(3, 93, 229, 0.7) 100%);
        width: fit-content;
    }
    </style>
""", unsafe_allow_html=True)



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

# 初始化狀態變量
if 'chatbot_api_key' not in st.session_state:
    st.session_state['chatbot_api_key'] = ''
if 'replicate_api_key' not in st.session_state:
    st.session_state['replicate_api_key'] = ''
if 'perplexity_api_key' not in st.session_state:
    st.session_state['perplexity_api_key'] = ''
if 'open_ai_model' not in st.session_state:
    st.session_state['open_ai_model'] = 'gpt-4o'
if 'perplexity_model' not in st.session_state:
    st.session_state['perplexity_model'] = 'llama-3-sonar-large-32k-chat'
if 'perplexity_temperature' not in st.session_state:
    st.session_state['perplexity_temperature'] = 0.5
if 'perplexity_top_p' not in st.session_state:
    st.session_state['perplexity_top_p'] = 0.5
if 'perplexity_presence_penalty' not in st.session_state:
    st.session_state['perplexity_presence_penalty'] = 0.0
if 'perplexity_length_penalty' not in st.session_state:
    st.session_state['perplexity_length_penalty'] = 0.0
if 'perplexity_max_tokens' not in st.session_state:
    st.session_state['perplexity_max_tokens'] = 1000
if 'perplexity_system_prompt' not in st.session_state:
    st.session_state['perplexity_system_prompt'] = ''
if 'gpt_system_prompt' not in st.session_state:
    st.session_state['gpt_system_prompt'] = ''
if 'language' not in st.session_state:
    st.session_state['language'] = ''
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.5
if 'top_p' not in st.session_state:
    st.session_state['top_p'] = 0.5
if 'presence_penalty' not in st.session_state:
    st.session_state['presence_penalty'] = 0.0
if 'frequency_penalty' not in st.session_state:
    st.session_state['frequency_penalty'] = 0.0
if 'max_tokens' not in st.session_state:
    st.session_state['max_tokens'] = 1000
if 'content' not in st.session_state:
    st.session_state['content'] = ''
if 'reset_confirmation' not in st.session_state:
    st.session_state['reset_confirmation'] = False
if 'tabs' not in st.session_state:
    st.session_state['tabs'] = ["對話 1"]
    st.session_state['current_tab'] = 1
    st.session_state[f"messages_ChatGPT_1"] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if not st.session_state['chatbot_api_key'] else "請問需要什麼協助？"}]
    st.session_state[f"messages_Perplexity_1"] = [{"role": "assistant", "content": "請輸入您的 Perplexity API Key" if not st.session_state['perplexity_api_key'] else "請問需要什麼協助？"}]
    st.session_state[f"tab_name_1"] = "對話 1"
if 'chat_started' not in st.session_state:
    st.session_state['chat_started'] = False
if 'api_key_removed' not in st.session_state:
    st.session_state['api_key_removed'] = False
if 'model_type' not in st.session_state:
    st.session_state['model_type'] = 'ChatGPT'

# 初始化 user_avatar 設定
if 'user_avatar_chatgpt' not in st.session_state:
    st.session_state['user_avatar_chatgpt'] = user_avatar_default
if 'user_avatar_perplexity' not in st.session_state:
    st.session_state['user_avatar_perplexity'] = user_avatar_default

# 根據模型選擇設置 user_avatar
if st.session_state['model_type'] == "ChatGPT":
    st.session_state['user_avatar'] = st.session_state['user_avatar_chatgpt']
else:
    st.session_state['user_avatar'] = st.session_state['user_avatar_perplexity']

def display_avatars():
    cols = st.columns(6)
    for i, (name, image) in enumerate(avatars.items()):
        with cols[i % 6]:
            st.image(f"data:image/png;base64,{image}", use_column_width=True)
            if st.button("選擇", key=name):
                if st.session_state['model_type'] == "ChatGPT":
                    st.session_state['user_avatar_chatgpt'] = image
                else:
                    st.session_state['user_avatar_perplexity'] = image
                st.session_state['user_avatar'] = image
                st.experimental_rerun()

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

def confirm_reset_chat():
    confirm, cancel = st.columns(2)
    with confirm:
        if st.button("確認", key="confirm_reset"):
            reset_chat()
            st.rerun()
    with cancel:
        if st.button("取消", key="cancel_reset"):
            st.session_state['reset_confirmation'] = False
            st.rerun()

def reset_chat():
    key = f"messages_{st.session_state['model_type']}_{st.session_state['current_tab']}"
    if st.session_state['model_type'] == 'ChatGPT':
        st.session_state[key] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if not st.session_state['chatbot_api_key'] else "請問需要什麼協助？"}]
    else:
        st.session_state[key] = [{"role": "assistant", "content": "請輸入您的 Perplexity API Key" if not st.session_state['perplexity_api_key'] else "請問需要什麼協助？"}]
    st.session_state['reset_confirmation'] = False
    st.session_state['chat_started'] = False
    st.session_state['api_key_removed'] = False

def update_model_params():
    st.session_state['temperature'] = st.session_state['temperature_slider']
    st.session_state['top_p'] = st.session_state['top_p_slider']
    st.session_state['presence_penalty'] = st.session_state['presence_penalty_slider']
    st.session_state['frequency_penalty'] = st.session_state['frequency_penalty_slider']

def format_message(text):
    return text.replace('\n', '<br>')

def update_gpt_system_prompt():
    st.session_state['gpt_system_prompt'] = st.session_state['gpt_system_prompt_input']

def update_perplexity_system_prompt():
    st.session_state['perplexity_system_prompt'] = st.session_state['perplexity_system_prompt_input']

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
                    <div class="message-container" style="background: {message_bg_color}; color: white; border-radius: 15px; padding: 10px; margin-right: 10px; font-size: 15px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                        {text} \n </div>
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
                    <div class="message-container" style="background: {message_bg_color}; color: #2B2727; border-radius: 15px; padding: 10px; margin-right: 5px; margin-left: 5px; font-size: 15px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                        {text} \n </div>
                </div>
                """,
            unsafe_allow_html=True,
        )

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
    st.write("\n")
    model_toggle = st.radio("", ["ChatGPT", "Perplexity"], key="model_type", horizontal=True)
    st.write("\n")
    # 根據模型選擇設置 avatar
    if model_toggle == "Perplexity":
        assistant_avatar = assistant_avatar_perplexity
        perplexity_api_key_input = st.text_input("請輸入 Perplexity API Key", value=st.session_state.get('perplexity_api_key', ''), type="password")
        if perplexity_api_key_input != st.session_state['perplexity_api_key']:
            st.session_state['perplexity_api_key'] = perplexity_api_key_input
            if not st.session_state['chat_started']:
                if perplexity_api_key_input:
                    st.session_state[f"messages_Perplexity_{st.session_state['current_tab']}"][0]['content'] = "請問需要什麼協助？"
                else:
                    st.session_state[f"messages_Perplexity_{st.session_state['current_tab']}"][0]['content'] = "請輸入您的 Perplexity API Key"
            st.experimental_rerun()
        
        if not st.session_state['chat_started']:
            st.session_state[f"messages_Perplexity_{st.session_state['current_tab']}"][0]['content'] = "請問需要什麼協助？" if perplexity_api_key_input else "請輸入您的 Perplexity API Key"
    else:
        assistant_avatar = assistant_avatar_gpt
        api_key_input = st.text_input("請輸入 OpenAI API Key", value=st.session_state.get('chatbot_api_key', ''), type="password")
        if api_key_input != st.session_state['chatbot_api_key']:
            st.session_state['chatbot_api_key'] = api_key_input
            if not st.session_state['chat_started']:
                st.session_state[f"messages_ChatGPT_{st.session_state['current_tab']}"][0]['content'] = "請問需要什麼協助？" if api_key_input else "請輸入您的 OpenAI API Key"
            st.experimental_rerun()

    st.divider()
    st.button("重置對話", on_click=lambda: st.session_state.update({'reset_confirmation': True}), use_container_width=True)
    if st.session_state.get('reset_confirmation', False):
        confirm_reset_chat()

    current_tab_key = f"messages_{st.session_state['model_type']}_{st.session_state['current_tab']}"
    if current_tab_key not in st.session_state:
        st.session_state[current_tab_key] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if st.session_state['model_type'] == "ChatGPT" and not st.session_state['chatbot_api_key'] else "請輸入您的 Perplexity API Key" if st.session_state['model_type'] == "Perplexity" and not st.session_state['perplexity_api_key'] else "請問需要什麼協助？"}]

if selected == "對話":
    current_tab_key = f"messages_{st.session_state['model_type']}_{st.session_state['current_tab']}"
    if current_tab_key not in st.session_state:
        st.session_state[current_tab_key] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if st.session_state['model_type'] == "ChatGPT" and not st.session_state['chatbot_api_key'] else "請輸入您的 Perplexity API Key" if st.session_state['model_type'] == "Perplexity" and not st.session_state['perplexity_api_key'] else "請問需要什麼協助？"}]

    # 顯示對話
    for msg in st.session_state[current_tab_key]:
        message_func(msg["content"], is_user=(msg["role"] == "user"))

    # 處理新訊息
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
                            <div class="message-container" style="background: #F1F1F1; color: 2B2727; border-radius: 15px; padding: 10px; margin-right: 5x; margin-left: 5px; font-size: 15px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                                {format_message(full_response)} \n </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.session_state[current_tab_key].append({"role": "assistant", "content": full_response})
                response_container.empty()
                message_func(full_response, is_user=False)

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
                    """
                    <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: flex-start;">
                        <img src="data:image/png;base64,{avatar}" class="bot-avatar" alt="avatar" style="width: 45px; height: 28px;" />
                        <div class="message-container" style="background: #F1F1F1; color: #2B2727; border-radius: 15px; padding: 10px; margin-right: 5px; margin-left: 5px; font-size: 15px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                            {content}
                        </div>
                    </div>
                    """.format(avatar=assistant_avatar_perplexity, content=full_response.replace('\n', '<br>')),
                    unsafe_allow_html=True
                )
    
            # 最後將完整的回應添加到會話狀態中
            st.session_state[current_tab_key].append({"role": "assistant", "content": full_response})
            response_container.empty()
            message_func(full_response, is_user=False)

elif selected == "模型設定":
    col1, col2, col3 = st.columns([2, 2, 1.5])
    if st.session_state['model_type'] == "ChatGPT":
        with col1:
            st.session_state['open_ai_model'] = st.selectbox("選擇 ChatGPT 模型", ["gpt-3.5-turbo", "gpt-4o"],
                                                             index=["gpt-3.5-turbo", "gpt-4o"].index(
                                                                 st.session_state.get('open_ai_model', 'gpt-4o')),
                                                             help="4：每百萬tokens = 20美元；3.5-turbo價格為其1/10")
        with col2:
            st.session_state['language'] = st.text_input("指定使用的語言", value=st.session_state.get('language'),
                                                         help="預設使用繁體中文。如要英文，請直接用中文輸入「英文」。")
        with col3:
            st.session_state['max_tokens'] = st.number_input("Tokens 上限", min_value=0,
                                                             value=st.session_state.get('max_tokens', 1000),
                                                             help="要生成的最大標記數量。")
        st.write("\n")
        st.text_area("角色設定", value=st.session_state.get('gpt_system_prompt', ''),
                     placeholder="你是一個友好且資深的英文老師。你的目標是幫助使用者提高他們的語言能力，並且用簡單易懂的方式解釋概念。你應該耐心回答問題，並鼓勵學生提出更多問題。",
                     help="用於給模型提供初始指導。", key="gpt_system_prompt_input", on_change=update_gpt_system_prompt)
        st.write("\n")
        with st.expander("模型參數", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.slider("選擇 Temperature", 
                          min_value=0.0, max_value=2.0, step=0.1, 
                          value=st.session_state['temperature'], 
                          help="較高的值會使輸出更隨機，而較低的值則會使其更加集中和確定性。一般建議只更改此參數或 Top P 中的一個，而不要同時更改。",
                          on_change=update_slider, args=(['temperature'],), kwargs={'value': st.session_state['temperature']})
                st.slider("選擇 Presence Penalty", 
                          min_value=-2.0, max_value=2.0, step=0.1, 
                          value=st.session_state['presence_penalty'], 
                          help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。",
                          on_change=update_slider, args=(['presence_penalty'],), kwargs={'value': st.session_state['presence_penalty']})
            with col2:
                st.slider("選擇 Top P", 
                          min_value=0.0, max_value=1.0, step=0.1, 
                          value=st.session_state['top_p'], 
                          help="基於核心機率的採樣，模型會考慮概率最高的top_p個標記的預測結果。當該參數為0.1時，代表只有包括前10%概率質量的標記將被考慮。一般建議只更改這個參數或 Temperature 中的一個，而不要同時更改。",
                          on_change=update_slider, args=(['top_p'],), kwargs={'value': st.session_state['top_p']})
                st.slider("選擇 Frequency Penalty", 
                          min_value=-2.0, max_value=2.0, step=0.1, 
                          value=st.session_state['frequency_penalty'], 
                          help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。",
                          on_change=update_slider, args=(['frequency_penalty'],), kwargs={'value': st.session_state['frequency_penalty']})
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
            st.session_state['language'] = st.text_input("指定使用的語言", value=st.session_state.get('language'), help="預設使用繁體中文。如要英文，請直接用中文輸入「英文」。")
        with col3:
            st.session_state['max_tokens'] = st.number_input("Tokens 上限", min_value=0, value=st.session_state.get('max_tokens', 1000), help="要生成的最大標記數量。")
        st.write("\n")
        st.text_area("角色設定", value=st.session_state.get('perplexity_system_prompt', ''),placeholder="你是一個專業的科技支援工程師。你的目標是幫助用戶解決各種技術問題，無論是硬體還是軟體問題。你應該詳細解釋解決方案，並確保用戶理解每一步驟。", help="用於給模型提供初始指導。", key="perplexity_system_prompt_input", on_change=update_perplexity_system_prompt)
        st.write("\n")
        with st.expander("模型參數",expanded=True):     
            col1, col2 = st.columns(2)
            with col1:
                st.slider("選擇 Temperature", 
                          min_value=0.0, max_value=2.0, step=0.1, 
                          value=st.session_state['temperature'], 
                          help="較高的值會使輸出更隨機，而較低的值則會使其更加集中和確定性。",
                          on_change=update_slider, args=(['temperature'],), kwargs={'value': st.session_state['temperature']})
            with col2:
                st.slider("選擇 Top P", 
                          min_value=0.0, max_value=1.0, step=0.1, 
                          value=st.session_state['top_p'], 
                          help="基於核心機率的採樣，模型會考慮概率最高的top_p個標記的預測結果。當該參數為0.1時，代表只有包括前10%概率質量的標記將被考慮。一般建議只更改這個參數或 Temperature 中的一個，而不要同時更改。",
                          on_change=update_slider, args=(['top_p'],), kwargs={'value': st.session_state['top_p']})
                
            st.slider("選擇 Presence Penalty", 
                          min_value=-2.0, max_value=2.0, step=0.1, 
                          value=st.session_state['presence_penalty'], 
                          help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。",
                          on_change=update_slider, args=(['presence_penalty'],), kwargs={'value': st.session_state['presence_penalty']})


elif selected == "提示詞":
    st.write("這是一個預留的空白頁面，用於將來的提示詞功能。")

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

