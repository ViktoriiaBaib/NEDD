import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from anthropic import Anthropic

# Configure the page
st.set_page_config(
    page_title="NEDD Server", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed"
)

# Custom CSS for better layout
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
    }
    .chat-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        height: 600px;
        overflow-y: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "anthropic_client" not in st.session_state:
    st.session_state.anthropic_client = None


def load_data():
    """Load data from Excel file"""
    try:
        df = pd.read_excel("data/data.xlsx")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def create_pairplot(df, selected_cols):
    """Create pairplot using seaborn"""
    try:
        # Create pairplot with seaborn
        plt.style.use("default")  # Ensure consistent style
        g = sns.pairplot(df[selected_cols], kind="scatter", corner=True, diag_kind="hist")
        g.fig.suptitle("Pairplot of Selected Features", y=1.02, fontsize=16)

        # Adjust layout and figure size
        g.fig.set_size_inches(10, 8)
        plt.tight_layout()

        return g.fig
    except Exception as e:
        st.error(f"Error creating pairplot: {str(e)}")
        return None


def initialize_claude():
    """Initialize Claude client"""
    api_key = st.session_state.get("claude_api_key", "")
    if api_key:
        try:
            client = Anthropic(api_key=api_key)
            st.session_state.anthropic_client = client
            return True
        except Exception as e:
            st.error(f"Error initializing Claude: {str(e)}")
            return False
    return False


def send_message_to_claude(message):
    """Send message to Claude and get response"""
    if not st.session_state.anthropic_client:
        return "Please enter your Claude API key first."

    try:
        response = st.session_state.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": message}],
        )
        return response.content[0].text
    except Exception as e:
        return f"Error communicating with Claude: {str(e)}"


# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🔬 NEDD Server</div>', unsafe_allow_html=True)

    # Load data
    df = load_data()

    if df is not None:
        # Create two main columns
        col1, col2 = st.columns([1, 1])

        with col1:
            # Data table section
            st.markdown('<div class="section-header">📊 Data Table</div>', unsafe_allow_html=True)
            st.dataframe(df, width="stretch", height=300)

            # Get first three columns for pairplot
            if len(df.columns) >= 3:
                selected_cols = df.columns[:3].tolist()
                st.markdown(
                    f'<div class="section-header">📈 Pairplot - Features: {", ".join(selected_cols)}</div>',
                    unsafe_allow_html=True,
                )

                # Create pairplot
                fig = create_pairplot(df, selected_cols)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close()
            else:
                st.warning("Dataset needs at least 3 columns for pairplot visualization.")

        with col2:
            # Claude chat section
            st.markdown('<div class="section-header">🤖 Claude Chat</div>', unsafe_allow_html=True)

            # API Key input
            if not st.session_state.get("claude_api_key"):
                api_key = st.text_input(
                    "Enter your Claude API Key:",
                    type="password",
                    help="Get your API key from https://console.anthropic.com/",
                )
                if st.button("Connect to Claude"):
                    if api_key:
                        st.session_state.claude_api_key = api_key
                        if initialize_claude():
                            st.success("Successfully connected to Claude!")
                            st.rerun()
                    else:
                        st.error("Please enter your API key.")
            else:
                # Chat interface
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)

                # Display chat history
                for role, message in st.session_state.chat_history:
                    if role == "user":
                        st.markdown(f"**You:** {message}")
                    else:
                        st.markdown(f"**Claude:** {message}")
                    st.markdown("---")

                st.markdown("</div>", unsafe_allow_html=True)

                # Chat input
                user_message = st.text_input("Ask Claude about your data:", key="chat_input")

                col_send, col_clear = st.columns([1, 1])
                with col_send:
                    if st.button("Send", width="stretch"):
                        if user_message:
                            # Add user message to history
                            st.session_state.chat_history.append(("user", user_message))

                            # Get Claude's response
                            with st.spinner("Claude is thinking..."):
                                response = send_message_to_claude(user_message)

                            # Add Claude's response to history
                            st.session_state.chat_history.append(("claude", response))

                            st.rerun()

                with col_clear:
                    if st.button("Clear Chat", width="stretch"):
                        st.session_state.chat_history = []
                        st.rerun()

                # Disconnect button
                if st.button("Disconnect Claude"):
                    st.session_state.claude_api_key = None
                    st.session_state.anthropic_client = None
                    st.session_state.chat_history = []
                    st.rerun()

    else:
        st.error("Could not load data. Please ensure 'data/data.xlsx' exists and is accessible.")

        # Show sample data structure for testing
        st.markdown("### Sample Data Structure")
        sample_data = {
            "Feature_1": np.random.randn(100),
            "Feature_2": np.random.randn(100),
            "Feature_3": np.random.randn(100),
            "Target": np.random.choice(["A", "B", "C"], 100),
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df.head(10))


if __name__ == "__main__":
    main()
