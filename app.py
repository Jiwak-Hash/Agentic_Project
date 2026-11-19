"""ShopAssist AI: a Streamlit view over the existing LangGraph workflow."""

import streamlit as st

from app_support import (ask_workflow, clear_history, get_history,
                         load_customer_ids)

EXAMPLES = [
    ("📦", "Where is order 1001?"),
    ("🚚", "What is your shipping policy?"),
    ("↩️", "Can I get a refund for order 1002?"),
    ("💬", "How can I contact customer support?"),
]

STYLE = """
<style>
  .stMainBlockContainer {max-width: 980px; padding-top: 4rem; padding-bottom: 2rem;}
  h1 {letter-spacing: -0.055em; font-weight: 750 !important;}
  [data-testid="stSidebar"] {border-right: 1px solid #dce9e8;}
  [data-testid="stSidebar"] .stButton button {border-radius: 10px;}
  [data-testid="stChatMessage"] {border: 1px solid #e0e9e9; border-radius: 16px;
    padding: 1.1rem; margin-bottom: .9rem;}
  [data-testid="stChatInput"] {border-radius: 16px;}
  .shop-eyebrow {font-size: .72rem; letter-spacing: .16em; font-weight: 700;
    color: #087f79; margin-bottom: .5rem;}
  .shop-welcome {background: linear-gradient(120deg, #e5f4f0, #eff6fa);
    border: 1px solid #d4e8e2; border-radius: 20px; padding: 1.7rem 1.9rem;
    margin: 1rem 0 1.5rem; color: #183c3a;}
  .shop-welcome h2 {font-size: 1.6rem; letter-spacing: -.035em; margin: 0 0 .4rem; padding: 0;}
  .shop-welcome p {margin: 0; color: #42625f; line-height: 1.7;}
  .stButton button {min-height: 3rem; text-align: left;}
  @media (max-width: 640px) {
    .stMainBlockContainer {padding-top: 3.5rem;}
    .shop-welcome {padding: 1.2rem;}
  }
</style>
"""


def show_message(message):
    avatar = ":material/shopping_bag:" if message["role"] == "assistant" else ":material/person:"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(f"Selected agent: {message['agent']}")
            with st.expander("Response details"):
                st.write(f"Agent: {message['agent']}")
                if message.get("intent"):
                    st.write("Intent: " + message["intent"].replace("_", " "))
                if message.get("sources"):
                    st.write("Policy sources: " + ", ".join(message["sources"]))
                else:
                    st.caption("No policy sources were returned for this response.")


def main():
    st.set_page_config(page_title="ShopAssist AI · Customer support",
                       page_icon="🛍️", layout="centered", initial_sidebar_state="auto")
    st.markdown(STYLE, unsafe_allow_html=True)  # Static styling only, never model text.
    with st.sidebar:
        st.markdown("### 🛍️ ShopAssist AI")
        st.caption("A little help for every order.")
        st.divider()
        st.subheader("Your demo account")
        st.caption("Demo only · This selector simulates a customer session. It is not real authentication.")
        try:
            customer_ids = load_customer_ids()
        except Exception:
            st.error("Demo customers couldn't be loaded. Check that the local order database is available.")
            st.stop()
        if not customer_ids:
            st.info("No demo customers are available in the order database.")
            st.stop()
        if st.session_state.get("demo_customer_id") not in customer_ids:
            st.session_state["demo_customer_id"] = 16 if 16 in customer_ids else customer_ids[0]
        customer_id = st.selectbox("Demo customer ID", customer_ids,
                                   format_func=lambda value: f"Customer {value}", key="demo_customer_id")
        st.caption("Each demo customer has a separate chat in this browser session.")
        if st.button("Clear this chat", icon=":material/delete_outline:", width="stretch", key="clear_chat"):
            clear_history(st.session_state, customer_id)
        st.divider()
        st.markdown("**Here to help with**")
        st.markdown("📦 Order tracking\n\n🚚 Shipping & return policies\n\n↩️ Refund eligibility")
        st.divider()
        st.caption("Semester project · Fictional store\n\nRefunds, order changes, and human escalation are demonstrations only.")

    st.markdown('<div class="shop-eyebrow">YOUR SHOPPING SUPPORT COMPANION</div>', unsafe_allow_html=True)
    st.title("ShopAssist AI")
    st.write("Order updates, clear policies, and a helpful next step.")
    st.caption(f"Chatting as Customer {customer_id} · Demo session")
    history = get_history(st.session_state, customer_id)
    if not history:
        st.markdown('<div class="shop-welcome"><h2>How can we help today?</h2>'
                    '<p>Ask about an order, explore shipping options, or check a return. '
                    'Choose a question below or write your own.</p></div>', unsafe_allow_html=True)

    example_prompt = None
    with st.expander("Try an example", expanded=not bool(history)):
        st.caption("Order examples depend on the selected demo customer. An order from another account will be unavailable. You can also ask ‘List my orders’ to find your own.")
        columns = st.columns(2)
        for index, (icon, question) in enumerate(EXAMPLES):
            with columns[index % 2]:
                if st.button(f"{icon}  {question}", key=f"example_{index}", width="stretch"):
                    example_prompt = question

    for message in history:
        show_message(message)
    st.caption("Include the order ID in each question. This demo handles one request at a time.")
    typed_prompt = st.chat_input("Ask about your order, shipping, or a refund…",
                                 key=f"chat_input_{customer_id}", max_chars=4000)
    prompt = typed_prompt or example_prompt
    if prompt and prompt.strip():
        history.append({"role": "user", "content": prompt})
        show_message(history[-1])
        with st.spinner("Finding the right help… The first policy question may take a little longer."):
            answer = ask_workflow(prompt, customer_id)
        history.append(answer)
        st.rerun()


if __name__ == "__main__":
    main()
