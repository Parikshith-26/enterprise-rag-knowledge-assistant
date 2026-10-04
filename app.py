import requests
import streamlit as st


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="ZX Bank Knowledge Assistant",
    page_icon="🏦",
    layout="wide",
)


# --------------------------------------------------
# FASTAPI BACKEND
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏦 ZX Bank Enterprise Knowledge Assistant")

st.caption(
    "AI-powered knowledge assistant for searching "
    "ZX Bank enterprise documents."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Assistant")

    st.write(
        "This assistant answers questions using "
        "the ZX Bank knowledge base."
    )

    st.divider()

    st.subheader("Knowledge Base")

    st.write("📚 ZX Bank")
    st.write("📄 Multi-format documents")
    st.write("🔎 Hybrid retrieval")
    st.write("🎯 Reranking")
    st.write("🧠 Query rewriting")
    st.write("🤖 Grounded generation")

    st.divider()

    # ----------------------------------------------
    # BACKEND HEALTH CHECK
    # ----------------------------------------------

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if health_response.status_code == 200:

            st.success("Backend: Online")

        else:

            st.error("Backend: Unavailable")

    except requests.RequestException:

        st.error("Backend: Offline")

    st.divider()

    # ----------------------------------------------
    # CLEAR CHAT
    # ----------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        # ------------------------------------------
        # DISPLAY SOURCES
        # ------------------------------------------

        if message["role"] == "assistant":

            sources = message.get(
                "sources",
                []
            )

            if sources:

                with st.expander(
                    f"📚 Sources ({len(sources)})"
                ):

                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):

                        st.markdown(
                            f"**{index}. "
                            f"{source['document_id']}**"
                        )

                        st.write(
                            f"**Section:** "
                            f"{source['section']}"
                        )

                        st.write(
                            f"**Chunk:** "
                            f"{source['chunk_id']}"
                        )

                        st.write(
                            f"**Document type:** "
                            f"{source['document_type']}"
                        )

                        st.write(
                            f"**Source:** "
                            f"{source['source']}"
                        )

                        if index < len(sources):

                            st.divider()


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about ZX Bank..."
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    # ----------------------------------------------
    # DISPLAY USER MESSAGE
    # ----------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    # ----------------------------------------------
    # SEND QUESTION TO FASTAPI
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the knowledge base..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/ask",

                    json={
                        "question": question,
                        "conversation_history": (
                            st.session_state.messages
                        ),
                    },

                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI "
                    "backend. Make sure Uvicorn is running."
                )

                st.stop()

            except requests.exceptions.Timeout:

                st.error(
                    "The backend took too long to respond."
                )

                st.stop()

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Backend request failed: {error}"
                )

                st.stop()

        # ------------------------------------------
        # ANSWER
        # ------------------------------------------

        answer = result.get(
            "answer",
            "No answer returned.",
        )

        st.markdown(answer)

        # ------------------------------------------
        # SOURCES
        # ------------------------------------------

        sources = result.get(
            "sources",
            []
        )

        if sources:

            with st.expander(
                f"📚 Sources ({len(sources)})"
            ):

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    st.markdown(
                        f"**{index}. "
                        f"{source['document_id']}**"
                    )

                    st.write(
                        f"**Section:** "
                        f"{source['section']}"
                    )

                    st.write(
                        f"**Chunk:** "
                        f"{source['chunk_id']}"
                    )

                    st.write(
                        f"**Document type:** "
                        f"{source['document_type']}"
                    )

                    st.write(
                        f"**Source:** "
                        f"{source['source']}"
                    )

                    if index < len(sources):

                        st.divider()

    # ----------------------------------------------
    # SAVE USER MESSAGE
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # ----------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )