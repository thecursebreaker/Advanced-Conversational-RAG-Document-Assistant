import streamlit as st
import json
from config import META_FILE, SHOW_QUERY_VARIANTS, SHOW_RETRIEVED_SOURCES
from llm_utils import get_llm
from memory import SessionMemory
from retrieval import (
    get_collection,
    handle_document_question,
    handle_corpus_summary_question,
)
from router import classify_route
from tools import list_files, answer_memory_question


@st.cache_resource
def init_system():
    llm = get_llm()
    collection = get_collection()
    return llm, collection


@st.cache_resource
def load_metadata():
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except:
        return {"files": []}


llm, collection = init_system()
metadata = load_metadata()


if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()

memory = st.session_state.memory


class DummyLogger:
    def info(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass

logger = DummyLogger()


st.title("📄 Local RAG Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.text_input("Ask a question about your documents:")

if question:
    route = classify_route(question)
    result = {}
    answer = ""

    if route == "document":
        result = handle_document_question(
            llm=llm,
            memory=memory,
            collection=collection,
            question=question,
            logger=logger,  
            metadata=metadata,
        )
        answer = result["answer"]

    elif route == "corpus_summary":
        result = handle_corpus_summary_question(
            llm=llm,
            memory=memory,
            collection=collection,
            question=question,
            logger=logger, 
            metadata=metadata,
        )
        answer = result["answer"]

    elif route == "meta":
        answer = list_files(metadata)

    elif route == "memory":
        answer = answer_memory_question(question, memory)
        
    elif route == "file_lookup":
        from tools import extract_file_position, get_file_by_position
        from retrieval import handle_file_summary_question

        position = extract_file_position(question)

        if position is None:
            answer = "I could not determine which document you meant."
        else:
            file_info = get_file_by_position(metadata, position)

            if file_info is None:
                answer = f"I could not find document {position}."
            else:
                result = handle_file_summary_question(
                    llm=llm,
                    filename=file_info["filename"],
                    collection=collection,
                    logger=logger,
                )
                answer = result["answer"]
    else:
        answer = "Unsupported route."

    memory.add_turn(question, answer)
    st.session_state.chat.append((question, answer))


for q, a in reversed(st.session_state.chat):
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)
    st.markdown("---")


if question:
    if SHOW_QUERY_VARIANTS and "query_variants" in result:
        st.subheader("Query variants")
        st.write(result["query_variants"])

    if SHOW_RETRIEVED_SOURCES and "results" in result:
        st.subheader("Sources")
        metas = result["results"]["metadatas"][0]
        for m in metas:
            st.write(m)