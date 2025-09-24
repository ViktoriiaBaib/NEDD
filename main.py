import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from anthropic import Anthropic
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def train_gpr_model(df):
    """Train Gaussian Process Regression model"""
    try:
        # Prepare features and target
        X = df.iloc[:, :-1].values  # All columns except last
        y = df.iloc[:, -1].values  # Last column as target

        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(df.iloc[:, -1]):
            st.error(
                "Target variable must be numeric for GPR. Please ensure the last column contains numeric values."
            )
            return None, None, None, None, None, None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define kernel and train GPR
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, random_state=42
        )
        gpr.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred, y_std = gpr.predict(X_test_scaled, return_std=True)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return gpr, scaler, X_test_scaled, y_test, y_pred, y_std, mse, r2
    except Exception as e:
        st.error(f"Error training GPR model: {str(e)}")
        return None, None, None, None, None, None, None, None


def plot_gpr_predictions(y_test, y_pred, y_std):
    """Plot GPR predictions with uncertainty bands"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort for better visualization
        sorted_indices = np.argsort(y_test)
        y_test_sorted = y_test[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        y_std_sorted = y_std[sorted_indices]

        # Plot true values
        ax.scatter(
            range(len(y_test_sorted)),
            y_test_sorted,
            alpha=0.7,
            color="blue",
            label="True Values",
            s=50,
        )

        # Plot predictions
        ax.scatter(
            range(len(y_pred_sorted)),
            y_pred_sorted,
            alpha=0.7,
            color="red",
            label="Predictions",
            s=50,
        )

        # Plot uncertainty band
        ax.fill_between(
            range(len(y_pred_sorted)),
            y_pred_sorted - 1.96 * y_std_sorted,
            y_pred_sorted + 1.96 * y_std_sorted,
            alpha=0.3,
            color="red",
            label="95% Confidence Interval",
        )

        ax.set_xlabel("Sample Index (sorted by true values)")
        ax.set_ylabel("Target Value")
        ax.set_title("GPR Predictions with Uncertainty")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig
    except Exception as e:
        st.error(f"Error creating GPR prediction plot: {str(e)}")
        return None


def uncertainty_sampling(gpr, scaler, X, n_samples=5):
    """Find points with highest model uncertainty"""
    try:
        X_scaled = scaler.transform(X)
        _, y_std = gpr.predict(X_scaled, return_std=True)

        # Get indices of highest uncertainty
        uncertain_indices = np.argsort(y_std)[-n_samples:][::-1]

        return uncertain_indices, y_std[uncertain_indices]
    except Exception as e:
        st.error(f"Error in uncertainty sampling: {str(e)}")
        return None, None


def expected_improvement(gpr, scaler, X, y_best, xi=0.01):
    """Calculate Expected Improvement for each point"""
    try:
        X_scaled = scaler.transform(X)
        y_pred, y_std = gpr.predict(X_scaled, return_std=True)

        # Calculate Expected Improvement
        improvement = y_pred - y_best - xi
        Z = improvement / y_std
        ei = improvement * norm.cdf(Z) + y_std * norm.pdf(Z)
        ei[y_std == 0.0] = 0.0

        return ei
    except Exception as e:
        st.error(f"Error calculating expected improvement: {str(e)}")
        return None


def plot_uncertainty_sampling(X, uncertain_indices, uncertainties, feature_names):
    """Plot uncertainty sampling results"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot all points
        ax.scatter(range(len(X)), np.arange(len(X)), alpha=0.3, color="gray", label="All Samples")

        # Highlight uncertain points
        ax.scatter(
            uncertain_indices,
            uncertain_indices,
            c=uncertainties,
            cmap="Reds",
            s=100,
            label="High Uncertainty Samples",
            edgecolors="black",
        )

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        ax.set_title("Uncertainty Sampling - Highest Model Uncertainty")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(ax.collections[-1], ax=ax)
        cbar.set_label("Uncertainty (σ)")

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating uncertainty sampling plot: {str(e)}")
        return None


def plot_expected_improvement(X, ei_values, n_samples=5):
    """Plot Expected Improvement results"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get top EI samples
        top_ei_indices = np.argsort(ei_values)[-n_samples:][::-1]

        # Plot all EI values
        ax.scatter(range(len(ei_values)), ei_values, alpha=0.5, color="blue", label="All Samples")

        # Highlight top EI points
        ax.scatter(
            top_ei_indices,
            ei_values[top_ei_indices],
            color="red",
            s=100,
            label="Top Expected Improvement",
            edgecolors="black",
            zorder=5,
        )

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Expected Improvement")
        ax.set_title("Expected Improvement - Best Candidates for Next Sampling")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig, top_ei_indices
    except Exception as e:
        st.error(f"Error creating expected improvement plot: {str(e)}")
        return None, None


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
            model="claude-3-5-sonnet-20241022",
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

            # Get all columns except the last one for pairplot
            if len(df.columns) >= 3:
                selected_cols = df.columns[:-1].tolist()
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

        # GPR Analysis Section (full width)
        st.markdown(
            '<div class="section-header">🤖 Gaussian Process Regression Analysis</div>',
            unsafe_allow_html=True,
        )

        # Check if target is numeric for GPR
        if pd.api.types.is_numeric_dtype(df.iloc[:, -1]):
            # Train GPR model
            with st.spinner("Training Gaussian Process Regression model..."):
                gpr_results = train_gpr_model(df)

            if gpr_results[0] is not None:  # Check if training was successful
                gpr, scaler, X_test_scaled, y_test, y_pred, y_std, mse, r2 = gpr_results

                # Display model performance
                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                with col_metrics2:
                    st.metric("R² Score", f"{r2:.4f}")

                # Create three columns for the plots
                col_plot1, col_plot2, col_plot3 = st.columns(3)

                with col_plot1:
                    st.markdown("**GPR Predictions with Uncertainty**")
                    fig_pred = plot_gpr_predictions(y_test, y_pred, y_std)
                    if fig_pred is not None:
                        st.pyplot(fig_pred)
                        plt.close()

                with col_plot2:
                    st.markdown("**Uncertainty Sampling**")
                    # Get full dataset for uncertainty analysis
                    X_full = df.iloc[:, :-1].values
                    feature_names = df.columns[:-1].tolist()

                    uncertain_indices, uncertainties = uncertainty_sampling(
                        gpr, scaler, X_full, n_samples=5
                    )
                    if uncertain_indices is not None:
                        fig_uncertainty = plot_uncertainty_sampling(
                            X_full, uncertain_indices, uncertainties, feature_names
                        )
                        if fig_uncertainty is not None:
                            st.pyplot(fig_uncertainty)
                            plt.close()

                        # Print suggested features for uncertainty sampling
                        st.markdown("**Highest Uncertainty Samples:**")
                        if uncertain_indices is not None and uncertainties is not None:
                            for i, (idx, uncertainty) in enumerate(
                                zip(uncertain_indices, uncertainties)
                            ):
                                feature_values = X_full[idx]
                                st.write(f"Sample {idx}: σ={uncertainty:.4f}")
                                feature_str = ", ".join(
                                    [
                                        f"{name}={val:.3f}"
                                        for name, val in zip(feature_names, feature_values)
                                    ]
                                )
                                st.write(f"Features: {feature_str}")

                with col_plot3:
                    st.markdown("**Expected Improvement**")
                    # Calculate expected improvement
                    y_best = float(np.max(np.array(df.iloc[:, -1])))  # Best observed value
                    ei_values = expected_improvement(gpr, scaler, X_full, y_best)

                    if ei_values is not None:
                        fig_ei, top_ei_indices = plot_expected_improvement(
                            X_full, ei_values, n_samples=5
                        )
                        if fig_ei is not None:
                            st.pyplot(fig_ei)
                            plt.close()

                        # Print suggested features for expected improvement
                        st.markdown("**Top Expected Improvement Samples:**")
                        if top_ei_indices is not None:
                            for i, idx in enumerate(top_ei_indices):
                                feature_values = X_full[idx]
                                ei_value = ei_values[idx]
                                st.write(f"Sample {idx}: EI={ei_value:.4f}")
                                feature_str = ", ".join(
                                    [
                                        f"{name}={val:.3f}"
                                        for name, val in zip(feature_names, feature_values)
                                    ]
                                )
                                st.write(f"Features: {feature_str}")
        else:
            st.warning(
                "GPR requires a numeric target variable. The last column should contain numeric values."
            )

        # Create second row for Claude chat
        st.markdown("---")
        col_chat = st.columns(1)[0]

        with col_chat:
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
