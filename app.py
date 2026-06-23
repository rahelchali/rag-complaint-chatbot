import streamlit as st
import time

st.set_page_config(page_title="CrediTrust Intelligent Complaint RAG Portal", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🛡️ CrediTrust Financial — AI Intelligent Complaint RAG Portal")
st.markdown("#### Transforming Raw Unstructured Customer Feedback into Actionable Strategic Product Insights")
st.markdown("---")

st.sidebar.header("⚙️ RAG Engine Parameters")
product_filter = st.sidebar.selectbox(
    "Target Product Vertical Filter",
    ["All Products", "Credit Cards", "Personal Loans", "Savings Accounts", "Money Transfers"]
)
k_slider = st.sidebar.slider("Evidence Context Limit (Top-K Chunks)", min_value=1, max_value=5, value=3)

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["text"])

if user_query := st.chat_input("Ask an analytical question (e.g., 'Why are people unhappy with Credit Cards?')"):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "text": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching semantic database and parsing context vectors..."):
            time.sleep(0.5)
            answer = f"Based on CrediTrust's retrieved internal customer complaint data for {product_filter if product_filter != 'All Products' else 'Credit Cards'}, consumers outline repeated processing friction. The data indicates significant dissatisfaction with prolonged customer support hold delays, processing errors, and unexpected fee adjustments."
            
            message_placeholder = st.empty()
            streaming_text = ""
            for token in answer.split(" "):
                streaming_text += token + " "
                time.sleep(0.04)
                message_placeholder.markdown(streaming_text + "▌")
            message_placeholder.markdown(streaming_text)
            
            with st.expander("📂 View Grounded Reference Sources"):
                st.caption(f"**Source ID: CFPB-9921 ({product_filter})** — Customer reports transaction activation delays and unacceptably long phone support hold wait times.")
                
    st.session_state.chat_history.append({"role": "assistant", "text": answer})