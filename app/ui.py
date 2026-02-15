# app/ui.py
import streamlit as st
from app.rag import ask

st.set_page_config(page_title="Security Policy Copilot", layout="wide")

st.title("🔐 Security Policy Copilot (Local RAG)")
st.caption(
    "Ask questions about NIST CSF 2.0, NIST SP 800-61r3, and OWASP Top 10 for LLM Apps — with citations."
)

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model", value="llama3.1:8b")

    audit_mode = st.checkbox("Audit mode (k=10)", value=False)
    if audit_mode:
        st.warning("Audit mode uses more context (k=10). More coverage, may include redundancy.")

    k = st.slider(
        "Top-k retrieved chunks",
        min_value=3,
        max_value=12,
        value=8,  # default
        step=1,
        disabled=audit_mode,
    )

    st.markdown("---")
    st.markdown("**Tip:** Default k=8 is a good balance. Use **Audit mode** when you want maximum coverage.")
    st.markdown("**Note:** You must have Ollama running locally.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_q = st.chat_input("Ask a security question (e.g., 'What are the CSF 2.0 Functions?')")

if user_q:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Run RAG + show assistant message
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating answer..."):
            out = ask(user_q, k=k, model=model, audit_mode=audit_mode)

        # Main answer
        st.markdown(out["answer"])

        # Right column: citations + retrieved context
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📌 Citations")
            # out["citations"] maps chunk -> file/page
            # We'll dedupe by (file,page) to keep it clean
            seen = set()
            cleaned = []
            for c in out.get("citations", []):
                key = (c.get("file"), c.get("page"))
                if key not in seen:
                    seen.add(key)
                    cleaned.append(c)

            if cleaned:
                for c in cleaned:
                    st.write(f"- **{c['file']}** p.{c['page']} (chunk [{c['chunk']}])")
            else:
                st.write("No citations returned.")

        with col2:
            st.subheader("🧾 Retrieved Context")
            retrieved = out.get("retrieved", [])
            if not retrieved:
                st.write("No retrieved chunks.")
            else:
                for r in retrieved:
                    label = f"[{r['chunk']}] {r['file']} p.{r['page']}"
                    with st.expander(label):
                        st.write(r["preview"])

    # Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": out["answer"]})
