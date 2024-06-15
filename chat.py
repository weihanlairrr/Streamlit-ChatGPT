import streamlit as st
import base64
from openai import OpenAI
from streamlit_option_menu import option_menu
import ollama  # 新增的匯入

# Custom CSS for slider color和button styles
st.markdown("""
    <style>
    .stButton > button {
        padding: 5px 20px;
        background-color: #e0e0e0;
        color: black;
        border: none;
        border-radius: 5px;
        font-size: 18px;
        cursor: pointer;
        margin: 1px 0; /* 僅設定上下邊距，消除左右邊距 */
        width: 100%; /* 讓按鈕自動佔滿整欄 */
    }
    .stButton > button:hover {
        background-color: #3399FF;
    }
    .bot-avatar.llama {
        width: 28px;
        height: 28px;
    }
    </style>
""", unsafe_allow_html=True)

def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 根據模型選擇載入對應的 avatar 圖片
assistant_avatar_gpt = get_image_as_base64("Images/ChatGPT Logo.png")
assistant_avatar_llama = get_image_as_base64("Images/Meta Logo.png")
assistant_avatar = assistant_avatar_gpt  # 默認為 GPT 模型的 avatar
user_avatar_default = get_image_as_base64("Images/Asian Man.png")

# 加載所有頭像圖片
avatars = {
    "Asian Man": get_image_as_base64("Images/Asian Man.png"),
    "Robot": get_image_as_base64("Images/Robot.png"),
    "Cat": get_image_as_base64("Images/Cat.png"),
    "Dog": get_image_as_base64("Images/Dog.png"),
    "Asian Bearded Man": get_image_as_base64("Images/Asian Bearded Man.png"),
    "Asian Boy": get_image_as_base64("Images/Asian Boy.png"),
    "Asian Woman": get_image_as_base64("Images/Asian Woman.png"),
    "Asian Rainbow Girl": get_image_as_base64("Images/Asian Rainbow Girl.png"),
    "Asian Girl": get_image_as_base64("Images/Asian Girl.png"),
}

# 預設使用者頭像
if 'user_avatar' not in st.session_state:
    st.session_state['user_avatar'] = user_avatar_default

# 初始化 session state 變數
if 'chatbot_api_key' not in st.session_state:
    st.session_state['chatbot_api_key'] = ''
if 'open_ai_model' not in st.session_state:
    st.session_state['open_ai_model'] = 'gpt-3.5-turbo'
if 'language' not in st.session_state:
    st.session_state['language'] = ''
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.5
if 'top_p' not in st.session_state:
    st.session_state['top_p'] = 1.0
if 'presence_penalty' not in st.session_state:
    st.session_state['presence_penalty'] = 0.0
if 'frequency_penalty' not in st.session_state:
    st.session_state['frequency_penalty'] = 0.0
if 'reset_confirmation' not in st.session_state:
    st.session_state['reset_confirmation'] = False
if 'tabs' not in st.session_state:
    st.session_state['tabs'] = ["對話 1"]
    st.session_state['current_tab'] = 1
    st.session_state[f"messages_ChatGPT_1"] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if not st.session_state['chatbot_api_key'] else "請問需要什麼協助？"}]
    st.session_state[f"messages_Llama3_1"] = [{"role": "assistant", "content": "請問需要什麼協助？"}]
    st.session_state[f"tab_name_1"] = "對話 1"
if 'chat_started' not in st.session_state:
    st.session_state['chat_started'] = False
if 'api_key_removed' not in st.session_state:
    st.session_state['api_key_removed'] = False
if 'model_type' not in st.session_state:
    st.session_state['model_type'] = 'ChatGPT'

# 定義一個函數來顯示頭像選項
def display_avatars():
    cols = st.columns(3)
    for i, (name, image) in enumerate(avatars.items()):
        with cols[i % 3]:
            st.image(f"data:image/png;base64,{image}", use_column_width=True)
            if st.button("選擇", key=name):
                st.session_state['user_avatar'] = image
                st.rerun()

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
        error_message = str(e)
        if "Incorrect API key provided" in error_message:
            return "請輸入正確的 OpenAI API Key"
        elif "insufficient_quota" in error_message:
            return "您的 OpenAI API餘額不足，請至您的帳戶加值"
        return f"Error: {error_message}"

def generate_ollama_response(prompt):
    try:
        response = ollama.chat(model='llama3', stream=True, messages=[{"role": "user", "content": prompt}])
        full_message = ""
        for partial_resp in response:
            token = partial_resp["message"]["content"]
            full_message += token
        return full_message
    except Exception as e:
        return f"Error: {str(e)}"

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
    st.session_state[key] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if st.session_state['model_type'] == 'ChatGPT' and not st.session_state['chatbot_api_key'] else "請問需要什麼協助？"}]
    st.session_state['reset_confirmation'] = False
    st.session_state['chat_started'] = False
    st.session_state['api_key_removed'] = False

def format_message(text):
    return text.replace('\n', '<br>')

def message_func(text, is_user=False, is_df=False):
    model_url = f"data:image/png;base64,{assistant_avatar}"
    user_url = f"data:image/png;base64,{st.session_state['user_avatar']}"

    avatar_url = model_url
    if is_user:
        avatar_url = user_url
        message_alignment = "flex-end"
        message_bg_color = "linear-gradient(135deg, #00B2FF 0%, #006AFF 100%)"
        avatar_class = "user-avatar"
        avatar_size = "width: 38px; height: 30;"
        st.markdown(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: {message_alignment};">
                    <div class="message-container" style="background: {message_bg_color}; color: white; border-radius: 10px; padding: 10px; margin-right: 5px; font-size: 17px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
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
        if assistant_avatar == assistant_avatar_llama:
            avatar_class += " llama"

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
                    <div class="message-container" style="background: {message_bg_color}; color: black; border-radius: 10px; padding: 10px; margin-right: 10px; margin-left: 5px; font-size: 17px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                        {text} \n </div>
                </div>
                """,
            unsafe_allow_html=True,
        )

with st.sidebar:
    model_toggle = st.radio("", ["ChatGPT", "Llama3"], key="model_type", horizontal=True)
    
    selected = option_menu("主頁", ["對話", '頭像', '模型設定'], 
        icons=['chat-left-dots', 'person-circle', 'gear'], menu_icon="cast", default_index=0,
        styles={
        "container": {"padding": "0.5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "22x"}, 
        "nav-link": {"font-size": "19px", "text-align": "left", "margin":"5px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#006AFF"}})
    
    # 根據模型選擇設置 avatar
    if model_toggle == "Llama3":
        assistant_avatar = assistant_avatar_llama
    else:
        assistant_avatar = assistant_avatar_gpt
    
    if selected == "對話":
        if model_toggle == "ChatGPT":
            api_key_input = st.text_input("請輸入 OpenAI API Key", value=st.session_state.get('chatbot_api_key', ''), type="password")
            if api_key_input != st.session_state['chatbot_api_key']:
                st.session_state['chatbot_api_key'] = api_key_input
                if api_key_input == '':
                    if not st.session_state['chat_started']:
                        reset_chat()
                    else:
                        st.session_state['api_key_removed'] = True
                else:
                    if not st.session_state['chat_started']:
                        st.session_state[f"messages_{st.session_state['model_type']}_{st.session_state['current_tab']}"][0]['content'] = "請問需要什麼協助？"
                    st.session_state['api_key_removed'] = False
                st.experimental_rerun()

            model_input = st.selectbox("選擇 GPT 模型", ("gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"), index=("gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o").index(st.session_state.get('open_ai_model', 'gpt-3.5-turbo')))
            if model_input != st.session_state['open_ai_model']:
                st.session_state['open_ai_model'] = model_input
                st.experimental_rerun()

        st.divider()
        st.button("重置對話", on_click=lambda: st.session_state.update({'reset_confirmation': True}), use_container_width=True)
        
        if st.session_state['reset_confirmation']:
            confirm_reset_chat()

    elif selected == "頭像":
        st.write("選擇您的頭像")
        display_avatars()
        st.write("\n")
        st.markdown(f"""
    <div style='text-align: center;'>
        <img src="data:image/png;base64,{st.session_state['user_avatar']}" style='width: 200px;'/>
        <p>\n</p>
    </div>
""", unsafe_allow_html=True)
            
        st.markdown("<p style='text-align: center;'>目前的頭像</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
            
    elif selected == "模型設定":
        st.session_state['language'] = st.text_input("指定使用的語言", value=st.session_state.get('language'), placeholder="預設為繁體中文")
        if model_toggle == "ChatGPT":
            st.session_state['temperature'] = st.select_slider("選擇 Temperature", options=[i/10.0 for i in range(11)], value=st.session_state.get('temperature', 0.5), help="較高的值會使輸出更隨機，而較低的值則會使其更加集中和確定性。一般建議只更改此參數或 Top P 中的一個，而不要同時更改。")
            st.session_state['top_p'] = st.select_slider("選擇 Top P", options=[i/10.0 for i in range(11)], value=st.session_state.get('top_p', 1.0), help="基於核心機率的採樣，模型會考慮概率最高的top_p個標記的預測結果。當該參數為0.1時，代表只有包括前10%概率質量的標記將被考慮。一般建議只更改這個參數或 Temperature 中的一個，而不要同時更改。")
            st.session_state['presence_penalty'] = st.select_slider("選擇 Presence Penalty", options=[i/10.0 for i in range(-20, 21)], value=st.session_state.get('presence_penalty', 0.0), help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。")
            st.session_state['frequency_penalty'] = st.select_slider("選擇 Frequency Penalty", options=[i/10.0 for i in range(-20, 21)], value=st.session_state.get('frequency_penalty', 0.0), help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。")

# 根據模型選擇設置當前對話
current_tab_key = f"messages_{st.session_state['model_type']}_{st.session_state['current_tab']}"
if current_tab_key not in st.session_state:
    st.session_state[current_tab_key] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if st.session_state['model_type'] == 'ChatGPT' and not st.session_state['chatbot_api_key'] else "請問需要什麼協助？"}]

# 顯示對話
for msg in st.session_state[current_tab_key]:
    message_func(msg["content"], is_user=(msg["role"] == "user"))

# 處理新訊息
if st.session_state['chatbot_api_key'] or st.session_state['model_type'] == "Llama3":
    prompt = st.chat_input()
    if prompt:
        st.session_state['chat_started'] = True
        if st.session_state['model_type'] == "ChatGPT" and not st.session_state['chatbot_api_key']:
            if st.session_state['api_key_removed'] or not st.session_state['chat_started']:
                st.session_state[current_tab_key].append({"role": "assistant", "content": "請輸入您的 OpenAI API Key"})
            st.experimental_rerun()
        else:
            st.session_state['api_key_removed'] = False
            if st.session_state['model_type'] == "ChatGPT":
                client = OpenAI(api_key=st.session_state['chatbot_api_key'])
                st.session_state[current_tab_key].append({"role": "user", "content": prompt})
                message_func(prompt, is_user=True)
        
                # 顯示 "Thinking..." 訊息
                thinking_placeholder = st.empty()
                st.session_state[current_tab_key].append({"role": "assistant", "content": "Thinking..."})
                with thinking_placeholder.container():
                    message_func("Thinking...", is_user=False)
                
                if st.session_state['language']:
                    prompt = prompt + f" 除非我要求翻譯，否則請完全使用{st.session_state['language']}回答"
                else:
                    prompt = prompt + f" 除非我要求翻譯，否則請使用繁體中文回答"
                messages = st.session_state[current_tab_key][:-1] + [{"role": "user", "content": prompt}]
        
                response_message = get_openai_response(client, st.session_state['open_ai_model'], messages, st.session_state['temperature'], st.session_state['top_p'], st.session_state['presence_penalty'], st.session_state['frequency_penalty'])
                
                # 清除 "Thinking..." 訊息並顯示真正的回應
                st.session_state[current_tab_key].pop()
                thinking_placeholder.empty()
                st.session_state[current_tab_key].append({"role": "assistant", "content": response_message})
                message_func(response_message, is_user=False)
            else:
                st.session_state[current_tab_key].append({"role": "user", "content": prompt})
                message_func(prompt, is_user=True)
        
                # 顯示 "Thinking..." 訊息
                thinking_placeholder = st.empty()
                st.session_state[current_tab_key].append({"role": "assistant", "content": "Thinking..."})
                with thinking_placeholder.container():
                    message_func("Thinking...", is_user=False)
                
                if st.session_state['language']:
                    prompt = prompt + f" 除非我要求翻譯，否則請完全使用{st.session_state['language']}回答。你無需說「我將使用{st.session_state['language']}回答」之類的話。"
                else:
                    prompt = prompt + f" 除非我要求翻譯，否則請使用繁體中文回答。你無需說「我將使用繁體中文回答」之類的話。"
        
                response_message = generate_ollama_response(prompt)
                
                # 清除 "Thinking..." 訊息並顯示真正的回應
                st.session_state[current_tab_key].pop()
                thinking_placeholder.empty()
                st.session_state[current_tab_key].append({"role": "assistant", "content": response_message})
                message_func(response_message, is_user=False)
