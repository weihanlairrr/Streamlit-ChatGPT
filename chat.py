import streamlit as st
import time
import base64
from openai import OpenAI
from streamlit_option_menu import option_menu

st.markdown("""
    <style>
    .stButton > button {
        padding: 10px 20px;
        background-color: #45a049;
        color: white;
        border: none;
        border-radius: 5px;
        font-size: 18px;
        cursor: pointer;
        margin: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

assistant_avatar = get_image_as_base64("ChatGPT logo.png")
user_avatar = get_image_as_base64("AI 3D Dog Avatar.png")

def get_openai_response(client, model, messages, temperature, top_p, presence_penalty, frequency_penalty):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def reset_chat():
    st.session_state[f"messages_{st.session_state['current_tab']}"] = [{"role": "assistant", "content": "請問需要什麼協助？"}]

def add_new_chat():
    base_name = "對話"
    new_tab_index = len(st.session_state['tabs']) + 1
    new_tab_name = f"{base_name} {new_tab_index}"
    
    while new_tab_name in st.session_state['tabs']:
        new_tab_index += 1
        new_tab_name = f"{base_name} {new_tab_index}"
    
    st.session_state['tabs'].append(new_tab_name)
    st.session_state[f"messages_{new_tab_index}"] = [{"role": "assistant", "content": "請問需要什麼協助？"}]
    st.session_state[f"tab_name_{new_tab_index}"] = new_tab_name
    st.session_state['current_tab'] = new_tab_index

def update_tab_name():
    current_tab_key = f"tab_name_{st.session_state['current_tab']}"
    
    if current_tab_key not in st.session_state:
        return

    new_name = st.session_state[current_tab_key]
    
    if new_name in st.session_state['tabs']:
        new_name_base = new_name
        count = 1
        while new_name in st.session_state['tabs']:
            new_name = f"{new_name_base}_{count}"
            count += 1
    
    st.session_state['tabs'][st.session_state['current_tab'] - 1] = new_name
    st.session_state[current_tab_key] = new_name

def delete_current_chat():
    if len(st.session_state['tabs']) == 1:
        placeholder = st.empty()
        placeholder.warning("無法刪除唯一的對話分頁。", icon="⚠️")
        time.sleep(2)
        placeholder.empty()
        return
    
    current_tab_index = st.session_state['current_tab']
    del st.session_state['tabs'][current_tab_index - 1]
    keys_to_delete = [key for key in st.session_state.keys() if key.startswith(f"messages_{current_tab_index}") or key.startswith(f"tab_name_{current_tab_index}")]
    for key in keys_to_delete:
        del st.session_state[key]
    
    if len(st.session_state['tabs']) == 0:
        st.session_state['current_tab'] = 1
    else:
        st.session_state['current_tab'] = min(current_tab_index, len(st.session_state['tabs']))
    
    if st.session_state['current_tab'] > len(st.session_state['tabs']):
        st.session_state['current_tab'] = len(st.session_state['tabs'])

if 'tabs' not in st.session_state:
    st.session_state['tabs'] = ["對話 1"]
    st.session_state['current_tab'] = 1
    st.session_state[f"messages_1"] = [{"role": "assistant", "content": "請問需要什麼協助？"}]
    st.session_state[f"tab_name_1"] = "對話 1"

if 'chatbot_api_key' not in st.session_state:
    st.session_state['chatbot_api_key'] = ''
if 'open_ai_model' not in st.session_state:
    st.session_state['open_ai_model'] = 'gpt-3.5-turbo'
if 'language' not in st.session_state:
    st.session_state['language'] = '繁體中文'
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 1.0
if 'top_p' not in st.session_state:
    st.session_state['top_p'] = 1.0
if 'presence_penalty' not in st.session_state:
    st.session_state['presence_penalty'] = 0.0
if 'frequency_penalty' not in st.session_state:
    st.session_state['frequency_penalty'] = 0.0

def format_message(text):
    return text.replace('\n', '<br>')

def message_func(text, is_user=False, is_df=False, model="gpt"):
    model_url = f"data:image/png;base64,{assistant_avatar}"
    user_url = f"data:image/png;base64,{user_avatar}"

    avatar_url = model_url
    if is_user:
        avatar_url = user_url
        message_alignment = "flex-end"
        message_bg_color = "linear-gradient(135deg, #00B2FF 0%, #006AFF 100%)"
        avatar_class = "user-avatar"
        avatar_size = "width: 38px; height: 30;"
        st.write(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px; justify-content: {message_alignment};">
                    <div class="message-container" style="background: {message_bg_color}; color: white; border-radius: 10px; padding: 10px; margin-right: 5px; font-size: 14px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
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
        avatar_size = "width: 50px; height: 28px;"

        if is_df:
            st.write(
                f"""
                    <div style="display: flex; align-items: center; margin-bottom: 20px; justify-content: {message_alignment};">
                        <img src="{model_url}" class="{avatar_class}" alt="avatar" style="{avatar_size}" />
                    </div>
                    """,
                unsafe_allow_html=True,
            )
            st.write(text)
            return
        else:
            text = format_message(text)

        st.write(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px; justify-content: {message_alignment};">
                    <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="{avatar_size}" />
                    <div class="message-container" style="background: {message_bg_color}; color: black; border-radius: 10px; padding: 10px; margin-right: 10px; margin-left: 5px; font-size: 14px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                        {text} \n </div>
                </div>
                """,
            unsafe_allow_html=True,
        )
        
current_tab_key = f"messages_{st.session_state['current_tab']}"
with st.sidebar:
    selected = option_menu("主頁", ["對話", '頭像','設定'], 
        icons=['chat-left-dots','person-circle' ,'gear'], menu_icon="cast", default_index=0,
        styles={
        "container": {"padding": "4!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"5px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"}})
    
    if selected == "對話":
        col1, col2 = st.columns(2)  
        with col1:
            st.button("新增對話", on_click=add_new_chat, use_container_width=True)
        with col2:
            st.button("刪除對話", on_click=delete_current_chat, use_container_width=True)
        st.text_input("修改對話名稱", key=f"tab_name_{st.session_state['current_tab']}", on_change=update_tab_name)
        
        st.divider()
        current_tab_index = st.session_state['current_tab'] - 1
        if current_tab_index >= len(st.session_state['tabs']):
            current_tab_index = len(st.session_state['tabs']) - 1
            st.session_state['current_tab'] = current_tab_index + 1

        selected_tab = st.radio(" ", st.session_state['tabs'], index=current_tab_index)
        st.session_state['current_tab'] = st.session_state['tabs'].index(selected_tab) + 1
        current_tab_key = f"messages_{st.session_state['current_tab']}"

    if selected == "設定":
        st.session_state['chatbot_api_key'] = st.text_input("請輸入 OpenAI API Key", value=st.session_state.get('chatbot_api_key', ''), type="password")
        with st.expander("模型設定"):
            st.session_state['open_ai_model'] = st.selectbox("選擇 GPT 模型", ("gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"), index=("gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o").index(st.session_state.get('open_ai_model', 'gpt-3.5-turbo')))
            st.session_state['language'] = st.text_input("指定使用的語言", value=st.session_state.get('language'), placeholder="預設為繁體中文")
            st.session_state['temperature'] = st.select_slider("選擇 Temperature", options=[i/10.0 for i in range(21)], value=st.session_state.get('temperature', 1.0))
            st.session_state['top_p'] = st.select_slider("選擇 Top P", options=[i/10.0 for i in range(11)], value=st.session_state.get('top_p', 1.0))
            st.session_state['presence_penalty'] = st.select_slider("選擇 Presence Penalty", options=[i/10.0 for i in range(-20, 21)], value=st.session_state.get('presence_penalty', 0.0))
            st.session_state['frequency_penalty'] = st.select_slider("選擇 Frequence Penalty", options=[i/10.0 for i in range(-20, 21)], value=st.session_state.get('frequency_penalty', 0.0))

current_tab_key = f"messages_{st.session_state['current_tab']}"
if current_tab_key not in st.session_state:
    st.session_state[current_tab_key] = [{"role": "assistant", "content": "請問需要什麼協助？"}]

for msg in st.session_state[current_tab_key]:
    message_func(msg["content"], is_user=(msg["role"] == "user"))

prompt = st.chat_input()
if prompt:
    if not st.session_state['chatbot_api_key']:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=st.session_state['chatbot_api_key'])
    st.session_state[current_tab_key].append({"role": "user", "content": prompt})
    message_func(prompt, is_user=True)

    # 顯示 "Thinking..." 訊息
    thinking_message = {"role": "assistant", "content": "Thinking..."}
    st.session_state[current_tab_key].append(thinking_message)
    thinking_placeholder = st.empty()
    with thinking_placeholder.container():
        message_func("Thinking...", is_user=False)
    
    if st.session_state['language']:
        prompt = prompt + f" 請完全使用{st.session_state['language']}回答"
    else:
        prompt = prompt + f" 請使用繁體中文回答"
    messages = st.session_state[current_tab_key][:-1] + [{"role": "user", "content": prompt}]

    response_message = get_openai_response(client, st.session_state['open_ai_model'], messages, st.session_state['temperature'], st.session_state['top_p'], st.session_state['presence_penalty'], st.session_state['frequency_penalty'])
    
    # 清除 "Thinking..." 訊息並顯示真正的回應
    st.session_state[current_tab_key].pop()
    thinking_placeholder.empty()
    st.session_state[current_tab_key].append({"role": "assistant", "content": response_message})
    message_func(response_message, is_user=False)
