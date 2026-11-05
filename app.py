import streamlit as st
from rag_backend import answer_question


st.set_page_config(page_title="Johor Election Chatbot", page_icon="🗳️")

st.title("🗳️ Johor State Election Chatbot")
st.write("Ask anything about the Johor state election 2026. "
         "Answers are based only on news articles from The Star.")

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    if len(sources) == 0:
        return
    st.markdown("**Sources:**")
    for s in sources:
        st.markdown("[" + str(s["number"]) + "] [" + s["title"] + "](" + s["url"] + ") — " + s["date"])


# show the old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg["sources"])


question = st.chat_input("Type your question here...")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching the articles..."):
            try:
                result = answer_question(question)
            except Exception as e:
                result = {
                    "answer": "Sorry, something went wrong. (" + str(e) + ")",
                    "sources": []
                }

        st.markdown(result["answer"])
        show_sources(result["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
