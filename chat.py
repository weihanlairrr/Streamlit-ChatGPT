from openai import OpenAI
import streamlit as st
import time
import base64

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
        avatar_size = "width: 45px; height: 33;"
        st.write(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px; justify-content: {message_alignment};">
                    <div class="message-container" style="background: {message_bg_color}; color: white; border-radius: 20px; padding: 10px; margin-right: 5px; font-size: 14px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                        {text} \n </div>
                    <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="{avatar_size}" />
                </div>
                """,
            unsafe_allow_html=True,
        )
    else:
        message_alignment = "flex-start"
        message_bg_color =  "linear-gradient(135deg, #909090 0%, #606060 100%)"
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
                    <div class="message-container" style="background: {message_bg_color}; color: white; border-radius: 20px; padding: 10px; margin-right: 10px; margin-left: 5px; font-size: 14px; max-width: 75%; word-wrap: break-word; word-break: break-all;">
                        {text} \n </div>
                </div>
                """,
            unsafe_allow_html=True,
        )

with st.sidebar:
    current_tab_index = st.session_state['current_tab'] - 1
    if current_tab_index >= len(st.session_state['tabs']):
        current_tab_index = len(st.session_state['tabs']) - 1
        st.session_state['current_tab'] = current_tab_index + 1

    selected_tab = st.radio(" ", st.session_state['tabs'], index=current_tab_index)
    st.session_state['current_tab'] = st.session_state['tabs'].index(selected_tab) + 1
    current_tab_key = f"tab_name_{st.session_state['current_tab']}"
    openai_api_key = st.text_input("請輸入 OpenAI API Key", key="chatbot_api_key", type="password")
    with st.expander("模型設定"):
        open_ai_model = st.selectbox("選擇 GPT 模型", ("gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"),help="3.5-turbo: US$1.5 / 1M tokens｜ 4-turbo價格為其20倍｜ 4o價格為其10倍")
        language = st.text_input("指定使用的語言",placeholder="預設為繁體中文")
        temperature = st.select_slider("選擇 Temperature", options=[i/10.0 for i in range(21)], value=1.0, help="較高的值（如0.8）會使輸出更隨機，而較低的值（如0.2）則會使其更加集中和確定性。建議僅更改 Top P或 Temperature 中的一個，而不要同時更改兩個。")
        top_p = st.select_slider("選擇 Top P", options=[i/10.0 for i in range(11)], value=1.0, help="當該參數設為0.1時，只有包含前10%概率質量的標記將被考慮。建議僅更改 Top P或 Temperature 中的一個，而不要同時更改兩個。")
        presence_penalty = st.select_slider("選擇 Presence Penalty", options=[i/10.0 for i in range(-20, 21)], value=0.0, help="該參數的取值範圍為-2.0到2.0。正值會根據新標記是否出現在當前生成的文本中對其進行懲罰，從而增加模型談論新話題的可能性。")
        frequency_penalty = st.select_slider("選擇 Frequence Penalty", options=[i/10.0 for i in range(-20, 21)], value=0.0, help="該參數的取值範圍為-2.0到2.0。正值會根據新標記在當前生成的文本中的已有頻率對其進行懲罰，從而減少模型直接重複相同語句的可能性。")
    st.write("\n")
    col1, col2, col3, col4 = st.columns([0.8,3,3,1.2])  
    with col2:
        st.button("新增對話", on_click=add_new_chat)
    with col3:
        st.button("刪除對話", on_click=delete_current_chat)
    
    st.text_input("修改對話名稱", key=current_tab_key, on_change=update_tab_name)

current_tab_key = f"messages_{st.session_state['current_tab']}"
if current_tab_key not in st.session_state:
    st.session_state[current_tab_key] = [{"role": "assistant", "content": "請問需要什麼協助？"}]

for msg in st.session_state[current_tab_key]:
    message_func(msg["content"], is_user=(msg["role"] == "user"))

prompt = st.chat_input()
if prompt:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    st.session_state[current_tab_key].append({"role": "user", "content": prompt})
    message_func(prompt, is_user=True)

    # 顯示 "Thinking..." 訊息
    thinking_message = {"role": "assistant", "content": "Thinking..."}
    st.session_state[current_tab_key].append(thinking_message)
    thinking_placeholder = st.empty()
    with thinking_placeholder.container():
        message_func("Thinking...", is_user=False)
    
    if language:
        prompt = prompt + f" 請完全使用{language}回答"
    else:
        prompt = prompt + f" 請使用繁體中文回答"
    messages = st.session_state[current_tab_key][:-1] + [{"role": "user", "content": prompt}]

    response_message = get_openai_response(client, open_ai_model, messages, temperature, top_p, presence_penalty, frequency_penalty)
    
    # 清除 "Thinking..." 訊息並顯示真正的回應
    st.session_state[current_tab_key].pop()
    thinking_placeholder.empty()
    st.session_state[current_tab_key].append({"role": "assistant", "content": response_message})
    message_func(response_message, is_user=False)
