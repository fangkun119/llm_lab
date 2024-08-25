import base64
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from PyPDF2 import PdfReader
from lib.web.msg_templates import bot_template, user_template
from lib.util.llm_utils import ChatModelUtil


def extract_text_from_pdf(files) -> str:
    text = ""
    for pdf in files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# def get_chat_chain(vector_store):
#    # 1. get chat model
#    llm = ChatModelUtil.getDefaultChatModel()
#    # 2. create conversation buffer memory
#    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#    # 3. create chat chain
#    conversation_chain = ConversationalRetrievalChain.from_llm(
#        llm=llm,
#        retriever=vector_store.as_retriever(),
#        memory=memory
#    )
#    return conversation_chain


def load_avatar(file_name):
    with open(file_name, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_image}"


def process_user_input(user_input):
    if st.session_state.rag_graph_wrapper is not None:
        # invoke st.session_state.conversation to get LLM response
        print("user input：" + user_input)
        response = st.session_state.conversation({'question': user_input})
        # response = st.session_state.rag_graph.get_graph().

        # streamlit session allows store different sessions for different users
        chat_history = response['chat_history']
        print("chat_history: " + str(chat_history))

        # load avatar
        user_avatar = load_avatar("static/user.png")
        ai_avatar = load_avatar("static/ai.png")

        # display chat_history
        for i, message in enumerate(chat_history):
            if i % 2 == 0:
                # user question
                st.write(
                    user_template
                        .replace("{{MSG}}", message.content)
                        .replace("{{USER_AVATAR}}", user_avatar),
                    unsafe_allow_html=True)  # unsafe_allow_html=True表示允许HTML内容被渲染
            else:
                # bot's answer
                st.write(
                    bot_template
                        .replace("{{MSG}}", message.content)
                        .replace("{{AI_AVATAR}}", ai_avatar),
                    unsafe_allow_html=True)
