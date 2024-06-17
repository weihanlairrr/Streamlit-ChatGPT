import streamlit as st
import base64
from streamlit_option_menu import option_menu
import replicate
import os
from openai import OpenAI


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
        margin: 5px 0;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #3399FF;
    }
    .stRadio {
        display: flex;
        justify-content: center;
        padding: 5px;
        border: none;
        border-radius: 5px;
        background-color: #e0e0e0;
        width: fit-content;
        margin: 1px 0;
    }
    </style>
""", unsafe_allow_html=True)

def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

assistant_avatar_gpt = get_image_as_base64("Images/ChatGPT Logo.png")
assistant_avatar_llama = get_image_as_base64("Images/Meta Logo.png")
user_avatar_default = get_image_as_base64("Images/Pink Bear.png")
logo_base64 = get_image_as_base64("Images/snow ai bot.png")

avatars = {
    "Pink Bear": get_image_as_base64("Images/Pink Bear.png"),
    "Cat": get_image_as_base64("Images/Cat.png"),
    "Dog": get_image_as_base64("Images/Dog.png"),
    "Robot": get_image_as_base64("Images/Robot.png"),
    "Asian Man": get_image_as_base64("Images/Asian Man.png"),
    "Asian Bearded Man": get_image_as_base64("Images/Asian Bearded Man.png"),
    "Asian Boy": get_image_as_base64("Images/Asian Boy.png"),
    "Asian Rainbow Girl": get_image_as_base64("Images/Asian Rainbow Girl.png"),
    "Asian Woman": get_image_as_base64("Images/Asian Woman.png"),
    "Asian Girl": get_image_as_base64("Images/Asian Girl.png"),
}

if 'chatbot_api_key' not in st.session_state:
    st.session_state['chatbot_api_key'] = ''
if 'replicate_api_key' not in st.session_state:
    st.session_state['replicate_api_key'] = ''
if 'open_ai_model' not in st.session_state:
    st.session_state['open_ai_model'] = 'gpt-3.5-turbo'
if 'llama_model' not in st.session_state:
    st.session_state['llama_model'] = 'meta/meta-llama-3-8b-instruct'
if 'llama_temperature' not in st.session_state:
    st.session_state['llama_temperature'] = 0.5
if 'llama_top_p' not in st.session_state:
    st.session_state['llama_top_p'] = 0.5
if 'llama_presence_penalty' not in st.session_state:
    st.session_state['llama_presence_penalty'] = 0.0
if 'llama_length_penalty' not in st.session_state:
    st.session_state['llama_length_penalty'] = 0.0
if 'llama_max_tokens' not in st.session_state:
    st.session_state['llama_max_tokens'] = 1000
if 'llama_system_prompt' not in st.session_state:
    st.session_state['llama_system_prompt'] = ''
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
    st.session_state[f"messages_Llama3_1"] = [{"role": "assistant", "content": "請輸入您的 Replicate API Key" if not st.session_state['replicate_api_key'] else "請問需要什麼協助？"}]
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
if 'user_avatar_llama3' not in st.session_state:
    st.session_state['user_avatar_llama3'] = user_avatar_default

# 根據模型選擇設置 user_avatar
if st.session_state['model_type'] == "ChatGPT":
    st.session_state['user_avatar'] = st.session_state['user_avatar_chatgpt']
else:
    st.session_state['user_avatar'] = st.session_state['user_avatar_llama3']

def display_avatars():
    cols = st.columns(6)
    for i, (name, image) in enumerate(avatars.items()):
        with cols[i % 6]:
            st.image(f"data:image/png;base64,{image}", use_column_width=True)
            if st.button("選擇", key=name):
                if st.session_state['model_type'] == "ChatGPT":
                    st.session_state['user_avatar_chatgpt'] = image
                else:
                    st.session_state['user_avatar_llama3'] = image
                st.session_state['user_avatar'] = image
                st.experimental_rerun()

def get_openai_response(client, model, messages, temperature, top_p, presence_penalty, frequency_penalty, max_tokens, system_prompt):
    try:
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        error_message = str(e)
        if "Incorrect API key provided" in error_message:
            return "請輸入正確的 OpenAI API Key"
        elif "insufficient_quota" in error_message:
            return "您的 OpenAI API餘額不足，請至您的帳戶加值"
        elif isinstance(e, UnicodeEncodeError):
            return "請輸入正確的 OpenAI API Key"
        return f"Error: {error_message}"


def generate_ollama_response(prompt, history, model, system_prompt):
    try:
        # 確保使用最新的 Replicate API key
        os.environ["REPLICATE_API_TOKEN"] = st.session_state['replicate_api_key']
        
        # 初始化 Replicate 客戶端
        client = replicate.Client(api_token=st.session_state['replicate_api_key'])
        
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        full_prompt = f"{system_prompt}\n\n{context}\nuser: {prompt}"
        
        response = client.run(
            model,
            input={"prompt": full_prompt, "temperature": st.session_state['llama_temperature'], "top_p": st.session_state['llama_top_p'], "presence_penalty": st.session_state['llama_presence_penalty'], "length_penalty": st.session_state['llama_length_penalty'], "max_tokens": st.session_state['llama_max_tokens']}
        )
        
        # 將 response 轉換為字符串並移除開頭多餘的換行
        if isinstance(response, list):
            response_str = ''.join(response)
        else:
            response_str = str(response)      
        # 移除開頭多餘的換行
        response_str = response_str.lstrip()
        
        return response_str
    except replicate.exceptions.ReplicateError as e:
        error_message = str(e)
        if "Unauthenticated" in error_message:
            return "請輸入正確的 Replicate API Key"
        return f"Error: {error_message}"
    except UnicodeEncodeError:
        return "請輸入正確的 Replicate API Key"

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
        st.session_state[key] = [{"role": "assistant", "content": "請輸入您的 Replicate API Key" if not st.session_state['replicate_api_key'] else "請問需要什麼協助？"}]
    st.session_state['reset_confirmation'] = False
    st.session_state['chat_started'] = False
    st.session_state['api_key_removed'] = False

def format_message(text):
    return text.replace('\n', '<br>')

def update_gpt_system_prompt():
    st.session_state['gpt_system_prompt'] = st.session_state['gpt_system_prompt_input']

def update_llama_system_prompt():
    st.session_state['llama_system_prompt'] = st.session_state['llama_system_prompt_input']


def message_func(text, is_user=False, is_df=False):
    model_url = f"data:image/png;base64,{assistant_avatar}"
    user_url = f"data:image/png;base64,{st.session_state['user_avatar']}"

    avatar_url = model_url
    if is_user:
        avatar_url = user_url
        message_alignment = "flex-end"
        message_bg_color = "linear-gradient(135deg, #00B2FF 0%, #006AFF 100%)"
        avatar_class = "user-avatar"
        avatar_size = "width: 30px; height: 30;"
        st.markdown(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 25px; justify-content: {message_alignment};">
                    <div class="message-container" style="background: {message_bg_color}; color: white; border-radius: 10px; padding: 10px; margin-right: 10px; font-size: 17px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
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
                    <div class="message-container" style="background: {message_bg_color}; color: black; border-radius: 10px; padding: 10px; margin-right: 10px; margin-left: 0px; font-size: 17px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
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
        icons=['chat-left-dots','gear','journal-text','robot'], menu_icon="robot", default_index=0,
        styles={
            "container": {"padding": "0.5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "22px"}, 
            "nav-link": {"font-size": "19px", "text-align": "left", "margin":"5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#006AFF"},
        }
    )
    
    model_toggle = st.radio("", ["ChatGPT", "Llama3"], key="model_type", horizontal=True)
    st.write("\n")
    # 根據模型選擇設置 avatar
    if model_toggle == "Llama3":
        assistant_avatar = assistant_avatar_llama
        replicate_api_key_input = st.text_input("請輸入 Replicate API Key", value=st.session_state.get('replicate_api_key', ''), type="password")
        if replicate_api_key_input != st.session_state['replicate_api_key']:
            st.session_state['replicate_api_key'] = replicate_api_key_input
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_key_input
            st.experimental_rerun()
        
        if not st.session_state['chat_started']:
            st.session_state[f"messages_Llama3_{st.session_state['current_tab']}"][0]['content'] = "請問需要什麼協助？" if replicate_api_key_input else "請輸入您的 Replicate API Key"
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

if selected == "對話":
    current_tab_key = f"messages_{st.session_state['model_type']}_{st.session_state['current_tab']}"
    if current_tab_key not in st.session_state:
        st.session_state[current_tab_key] = [{"role": "assistant", "content": "請輸入您的 OpenAI API Key" if st.session_state['model_type'] == "ChatGPT" and not st.session_state['chatbot_api_key'] else "請輸入您的 Replicate API Key" if st.session_state['model_type'] == "Llama3" and not st.session_state['replicate_api_key'] else "請問需要什麼協助？"}]

    # 顯示對話
    for msg in st.session_state[current_tab_key]:
        message_func(msg["content"], is_user=(msg["role"] == "user"))

    # 處理新訊息
    if st.session_state['model_type'] == "ChatGPT" and st.session_state['chatbot_api_key']:
        prompt = st.chat_input()
        if prompt:
            st.session_state['chat_started'] = True
            client = OpenAI(api_key=st.session_state['chatbot_api_key'])
            st.session_state[current_tab_key].append({"role": "user", "content": prompt})
            message_func(prompt, is_user=True)
            
            # 顯示 "Thinking..." 訊息
            thinking_placeholder = st.empty()
            st.session_state[current_tab_key].append({"role": "assistant", "content": "Thinking..."})
            with thinking_placeholder.container():
                message_func("Thinking...", is_user=False)
            
            if st.session_state['language']:
                prompt = prompt + f" 除非我要求翻譯，否則請完全使用{st.session_state['language']}回答。你無需說「明白了我將使用{st.session_state['language']}回答」或「好的」之類的話。"

            messages = st.session_state[current_tab_key][:-1] + [{"role": "user", "content": prompt}]
            
            response_message = get_openai_response(client, st.session_state['open_ai_model'], messages, st.session_state['temperature'], st.session_state['top_p'], st.session_state['presence_penalty'], st.session_state['frequency_penalty'], st.session_state['max_tokens'], st.session_state['gpt_system_prompt'])
            
            # 清除 "Thinking..." 訊息並顯示真正的回應
            st.session_state[current_tab_key].pop()
            thinking_placeholder.empty()
            st.session_state[current_tab_key].append({"role": "assistant", "content": response_message})
            message_func(response_message, is_user=False)

    elif st.session_state['model_type'] == "Llama3" and st.session_state['replicate_api_key']:
        prompt = st.chat_input()
        if prompt:
            st.session_state['chat_started'] = True
            st.session_state[current_tab_key].append({"role": "user", "content": prompt})
            message_func(prompt, is_user=True)
            
            # 顯示 "Thinking..." 訊息
            thinking_placeholder = st.empty()
            st.session_state[current_tab_key].append({"role": "assistant", "content": "Thinking..."})
            with thinking_placeholder.container():
                message_func("Thinking...", is_user=False)
            
            if st.session_state['language']:
                prompt = prompt + f" 除非我要求翻譯，否則請完全使用{st.session_state['language']}回答。你無需說「明白了我將使用{st.session_state['language']}回答」或「好的」之類的話。"
            else:
                prompt = prompt + f" 除非我要求翻譯，否則請完全使用繁體中文回答。你無需說「明白了我將使用繁體中文回答」或「好的」之類的話。"
            
            response_message = generate_ollama_response(prompt, st.session_state[current_tab_key], st.session_state['llama_model'], st.session_state['llama_system_prompt'])
            
            # 清除 "Thinking..." 訊息並顯示真正的回應
            st.session_state[current_tab_key].pop()
            thinking_placeholder.empty()
            st.session_state[current_tab_key].append({"role": "assistant", "content": response_message})
            message_func(response_message, is_user=False)

elif selected == "模型設定":
    col1, col2 ,col3 = st.columns([2,2,1])
    if st.session_state['model_type'] == "ChatGPT":
        with col1:
            st.session_state['open_ai_model'] = st.selectbox("選擇 ChatGPT 模型", ["gpt-3.5-turbo", "gpt-4o"], index=["gpt-3.5-turbo", "gpt-4o"].index(st.session_state.get('open_ai_model', 'gpt-3.5-turbo')),help="4o：每百萬tokens = 20美元；3.5-turbo價格為其1/10")
        with col2:
            st.session_state['language'] = st.text_input("指定使用的語言", value=st.session_state.get('language'), help="預設使用繁體中文。如要英文，請直接用中文輸入「英文」。")
        with col3:
            st.session_state['max_tokens'] = st.number_input("Tokens 上限", min_value=0, value=st.session_state.get('max_tokens', 1000), help="要生成的最大標記數量。")
        st.text_area("角色設定", value=st.session_state.get('gpt_system_prompt', ''), placeholder="你是一個友好且資深的英文老師。你的目標是幫助使用者提高他們的語言能力，並且用簡單易懂的方式解釋概念。你應該耐心回答問題，並鼓勵學生提出更多問題。",help="用於給模型提供初始指導。", key="gpt_system_prompt_input", on_change=update_gpt_system_prompt)

        st.write("\n")
        with st.expander("模型參數",expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state['temperature'] = st.select_slider("選擇 Temperature", options=[i/10.0 for i in range(11)], value=st.session_state.get('temperature', 0.5), help="較高的值會使輸出更隨機，而較低的值則會使其更加集中和確定性。一般建議只更改此參數或 Top P 中的一個，而不要同時更改。")
                st.session_state['presence_penalty'] = st.select_slider("選擇 Presence Penalty", options=[i/10.0 for i in range(-20, 21)], value=st.session_state.get('presence_penalty', 0.0), help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。")
            with col2:
                st.session_state['top_p'] = st.select_slider("選擇 Top P", options=[i/10.0 for i in range(11)], value=st.session_state.get('top_p', 1.0), help="基於核心機率的採樣，模型會考慮概率最高的top_p個標記的預測結果。當該參數為0.1時，代表只有包括前10%概率質量的標記將被考慮。一般建議只更改這個參數或 Temperature 中的一個，而不要同時更改。")
                st.session_state['frequency_penalty'] = st.select_slider("選擇 Frequency Penalty", options=[i/10.0 for i in range(-20, 21)], value=st.session_state.get('frequency_penalty', 0.0), help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。")
            
    else:
        with col1:
            # 定義模型名稱的映射
            llama_model_options = {
                "llama-3-8b-instruct": "meta/meta-llama-3-8b-instruct",
                "llama-3-70b-instruct": "meta/meta-llama-3-70b-instruct"
            }
            # 顯示簡化後的模型名稱
            selected_model = st.selectbox("選擇 Llama3 模型", list(llama_model_options.keys()),help="70b-instruct：每百萬tokens = 2.75美元；8b-instruct：每百萬tokens = 0.25美元")

            # 將選擇的簡化名稱映射到完整名稱
            st.session_state['llama_model'] = llama_model_options[selected_model]
        with col2:
            st.session_state['language'] = st.text_input("指定使用的語言", value=st.session_state.get('language'), help="預設使用繁體中文。如要英文，請直接用中文輸入「英文」。")
        with col3:
            st.session_state['llama_max_tokens'] = st.number_input("Tokens 上限", min_value=0, value=st.session_state.get('llama_max_tokens', 1000), help="要生成的最大標記數量。")
        st.text_area("角色設定", value=st.session_state.get('llama_system_prompt', ''),placeholder="你是一個專業的科技支援工程師。你的目標是幫助用戶解決各種技術問題，無論是硬體還是軟體問題。你應該詳細解釋解決方案，並確保用戶理解每一步驟。", help="用於給模型提供初始指導。", key="llama_system_prompt_input", on_change=update_llama_system_prompt)
        st.write("\n")
        with st.expander("模型參數",expanded=True):     
            col1, col2 = st.columns(2)
            with col1:
                st.session_state['llama_temperature'] = st.select_slider("選擇 Temperature", options=[i/10.0 for i in range(11)], value=st.session_state.get('llama_temperature', 0.5), help="較高的值會使輸出更隨機，而較低的值則會使其更加集中和確定性。")
                st.session_state['llama_presence_penalty'] = st.select_slider("選擇 Presence Penalty", options=[i/10.0 for i in range(-20, 21)], value=st.session_state.get('llama_presence_penalty', 0.0), help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。")
            with col2:
                st.session_state['llama_top_p'] = st.select_slider("選擇 Top P", options=[i/10.0 for i in range(11)], value=st.session_state.get('llama_top_p', 1.0), help="基於核心機率的採樣，模型會考慮概率最高的top_p個標記的預測結果。當該參數為0.1時，代表只有包括前10%概率質量的標記將被考慮。一般建議只更改這個參數或 Temperature 中的一個，而不要同時更改。")
                st.session_state['llama_length_penalty'] = st.select_slider("選擇 Length Penalty", options=[i/10.0 for i in range(-20, 21)], value=st.session_state.get('llama_length_penalty', 1.0), help="正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。")

elif selected == "提示詞":

    st.write("這是一個預留的空白頁面，用於將來的提示詞功能。")

elif selected == "頭像":
    st.markdown(f"""
        <div style='text-align: center;'>
            <div style='display: inline-block; border-radius: 60%; overflow: hidden; border: 7px solid #3399FF;'>
                <img src="data:image/png;base64,{st.session_state['user_avatar']}" style='width: 150px;'/>
            </div>
            <p>\n</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #999;'>目前的頭像</p>", unsafe_allow_html=True)
    st.write("\n")
    display_avatars()
