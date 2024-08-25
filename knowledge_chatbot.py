import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lib.util.pdf_utils import extract_text_from_pdf
from lib.util.pdf_utils import process_user_input
from lib.util.pdf_utils import get_chat_chain
from lib.config.environment import Environment
from lib.config.app_config import Config
from lib.util.llm_utils import EmbeddingUtil
from lib.util.vector_store_utils import VectorStoreUtil

Environment.setup_up_env_vars(enable_langsmith=False, langsmith_proj="rag_bot")


def load_or_init_session():
    # init session variables
    print("load or init chat session")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore_wrapper" not in st.session_state:
        st.session_state.vectorstore_wrapper = None

    # init vector store, conversation, chat_history
    if st.session_state.vectorstore_wrapper is None:
        print("init knowledge base for new chat")
        vectorstore_wrapper = VectorStoreUtil.create_default_vectorstore_wrapper()
        vectorstore_wrapper.init_from_dump(
            embedding=EmbeddingUtil.getDefaultEmbeddingModel(),
            base_dir=Config.vectorstore_dump_dir())
        st.session_state.vectorstore_wrapper = vectorstore_wrapper
    if st.session_state.conversation is None:
        print("init conversation for new chat")
        st.session_state.conversation = get_chat_chain(st.session_state.vectorstore_wrapper.get_vector_store())


def add_text_into_vectorstore(texts: str):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=0)
    doc_splits = splitter.create_documents(splitter.split_text(texts))
    # 6. vectorize and save into vector store
    st.session_state.vectorstore_wrapper.add_docs(doc_splits, EmbeddingUtil.getDefaultEmbeddingModel())


def main():
    # web page config
    st.set_page_config(page_title="ReadMind", page_icon=":robot:")
    st.header("ReadMind：Your AI DOC Assistant")

    # load or init session
    load_or_init_session()

    # web page component
    # 1. text box that process user input
    user_input = st.text_input("Input your question: ")
    if user_input:
        process_user_input(user_input)

    # 2. sidebar
    with st.sidebar:
        # sidebar header
        st.subheader("Add DOC to knowledge base")
        # button that upload file into knowledge base
        files = st.file_uploader("upload your document，then click 'Submit'", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("processing ..."):
                texts = extract_text_from_pdf(files)
                add_text_into_vectorstore(texts)


if __name__ == "__main__":
    main()
