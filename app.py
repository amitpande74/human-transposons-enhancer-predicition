import streamlit as st
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Enhancer Detection Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the pre-trained enhancer detection model"""
    try:
        model = tf.keras.models.load_model('best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
        
# One-hot encoding function
def one_hot_encode(sequence):
    """Convert DNA sequence to one-hot encoded format"""
    encoding = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0.25, 0.25, 0.25, 0.25]
    }
    return np.array([encoding.get(base.upper(), [0.25, 0.25, 0.25, 0.25]) for base in sequence], dtype=np.float32)

def process_sequences(sequences, max_length):
    """Process sequences for model prediction"""
    encoded_seqs = [one_hot_encode(seq) for seq in sequences]
    padded_seqs = pad_sequences(encoded_seqs, maxlen=max_length, padding='post')
    return np.array(padded_seqs)

def predict_enhancers(model, sequences, max_length):
    """Predict enhancer probability for given sequences"""
    processed_seqs = process_sequences(sequences, max_length)
    predictions = model.predict(processed_seqs)
    return predictions.flatten()

def parse_fasta_content(content):
    """Parse FASTA content and return sequences with their IDs"""
    sequences = []
    sequence_ids = []
    
    try:
        # Handle both string and bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Create a StringIO object to simulate a file
        fasta_io = io.StringIO(content)
        
        # Parse sequences
        for record in SeqIO.parse(fasta_io, "fasta"):
            sequences.append(str(record.seq).upper())
            sequence_ids.append(record.id)
            
        return sequences, sequence_ids
    except Exception as e:
        st.error(f"Error parsing FASTA file: {str(e)}")
        return [], []

def classify_enhancer_strength(probability):
    """Classify enhancer strength based on probability"""
    if probability >= 0.999:
        return "Very Strong"
    elif probability >= 0.99:
        return "Strong"
    elif probability >= 0.95:
        return "Moderate"
    elif probability >= 0.9:
        return "Weak"
    else:
        return "Very Weak"

# Main app
def main():
    st.title("🧬 DNA Enhancer Detection Tool")
    st.markdown("---")
    
    # Critical warning about model limitations
    st.error("""
    🚨 **Critical Model Limitation**: 
    This model currently predicts most sequences as enhancers, including clear negative controls.
    Results should NOT be used for scientific conclusions without experimental validation.
    This tool is for demonstration and screening purposes only.
    """)
    
    # Sidebar
    st.sidebar.header("About This Tool")
    st.sidebar.info(
        """
        This tool uses a deep learning model (1D CNN) to predict whether 
        DNA sequences are enhancers. Upload your sequences in FASTA format 
        or paste them directly to get predictions.
        """
    )
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        """
        1. Choose input method (upload file or paste text)
        2. Provide DNA sequences in FASTA format
        3. Click 'Predict Enhancers' to get results
        4. Interpret results with caution (see limitations)
        5. Download results as CSV
        """
    )
    
    # Enhanced performance warning in sidebar
    st.sidebar.error("""
    **Known Model Issues**: 
    • High false positive rate
    • Predicts most sequences as enhancers
    • Trained on imbalanced dataset
    • Requires experimental validation
    """)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Set max_length based on your training (you may need to adjust this)
    MAX_LENGTH = 1000  # Adjust based on your model's training
    
    # Input methods
    st.header("Input DNA Sequences")
    input_method = st.radio(
        "Choose input method:",
        ["Upload FASTA file", "Paste sequences"]
    )
    
    sequences = []
    sequence_ids = []
    
    if input_method == "Upload FASTA file":
        uploaded_file = st.file_uploader(
            "Upload FASTA file",
            type=['fasta', 'fa', 'fas', 'txt'],
            help="Upload a FASTA file containing DNA sequences"
        )
        
        if uploaded_file is not None:
            content = uploaded_file.read()
            sequences, sequence_ids = parse_fasta_content(content)
            st.success(f"Loaded {len(sequences)} sequences from file")
            
    else:  # Paste sequences
        fasta_text = st.text_area(
            "Paste FASTA sequences here:",
            height=200,
            placeholder=">sequence1\nATCGATCGATCG...\n>sequence2\nGCTAGCTAGCTA..."
        )
        
        if fasta_text:
            sequences, sequence_ids = parse_fasta_content(fasta_text)
            if sequences:
                st.success(f"Parsed {len(sequences)} sequences")
    
    # Display sequence info
    if sequences:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of sequences", len(sequences))
        with col2:
            st.metric("Average length", f"{np.mean([len(seq) for seq in sequences]):.0f} bp")
        with col3:
            st.metric("Max length", f"{max([len(seq) for seq in sequences])} bp")
        
        # Show first few sequences
        with st.expander("Preview sequences"):
            preview_df = pd.DataFrame({
                'ID': sequence_ids[:5],
                'Length': [len(seq) for seq in sequences[:5]],
                'First 50 bp': [seq[:50] + '...' if len(seq) > 50 else seq for seq in sequences[:5]]
            })
            st.dataframe(preview_df)
    
    # Prediction
    if sequences and st.button("🔬 Predict Enhancers", type="primary"):
        with st.spinner("Analyzing sequences..."):
            try:
                # Make predictions
                predictions = predict_enhancers(model, sequences, MAX_LENGTH)
                
                # Create results dataframe with multiple classification schemes
                results_df = pd.DataFrame({
                    'Sequence_ID': sequence_ids,
                    'Sequence_Length': [len(seq) for seq in sequences],
                    'Enhancer_Probability': predictions,
                    'Binary_Prediction_999': ['Enhancer' if p > 0.999 else 'Non-enhancer' for p in predictions],
                    'Binary_Prediction_99': ['Enhancer' if p > 0.99 else 'Non-enhancer' for p in predictions],
                    'Binary_Prediction_95': ['Enhancer' if p > 0.95 else 'Non-enhancer' for p in predictions],
                    'Enhancer_Strength': [classify_enhancer_strength(p) for p in predictions],
                    'Confidence': ['High' if abs(p - 0.5) > 0.3 else 'Medium' if abs(p - 0.5) > 0.1 else 'Low' for p in predictions]
                })
                
                # Interpretation guide
                st.info("""
                💡 **How to Interpret Results**: 
                Due to model limitations, focus on **relative differences** between sequences:
                • **Very Strong** (≥0.999): Highest model confidence
                • **Strong** (≥0.99): High model confidence  
                • **Moderate** (≥0.95): Moderate confidence
                • **Weak** (≥0.9): Lower confidence
                • **Very Weak** (<0.9): Lowest confidence (rare with this model)
                
                🔬 **Important**: All predictions require experimental validation.
                """)
                
                # Display results
                st.header("📊 Results")
                
                # Summary statistics with multiple thresholds
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total sequences", len(results_df))
                with col2:
                    enhancers_999 = (results_df['Enhancer_Probability'] > 0.999).sum()
                    st.metric("Very Strong (>0.999)", enhancers_999)
                with col3:
                    enhancers_99 = (results_df['Enhancer_Probability'] > 0.99).sum()
                    st.metric("Strong+ (>0.99)", enhancers_99)
                with col4:
                    st.metric("Average probability", f"{results_df['Enhancer_Probability'].mean():.3f}")
                
                # Enhanced visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Probability distribution with multiple threshold lines
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(results_df['Enhancer_Probability'], bins=20, alpha=0.7, edgecolor='black')
                    ax.axvline(x=0.999, color='red', linestyle='--', label='Very Strong (0.999)', linewidth=2)
                    ax.axvline(x=0.99, color='orange', linestyle='--', label='Strong (0.99)', linewidth=2)
                    ax.axvline(x=0.95, color='yellow', linestyle='--', label='Moderate (0.95)', linewidth=2)
                    ax.axvline(x=0.9, color='green', linestyle='--', label='Weak (0.9)', linewidth=2)
                    ax.set_xlabel('Enhancer Probability')
                    ax.set_ylabel('Number of Sequences')
                    ax.set_title('Distribution of Enhancer Probabilities')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # Enhancer strength distribution
                    fig, ax = plt.subplots(figsize=(10, 6))
                    strength_counts = results_df['Enhancer_Strength'].value_counts()
                    colors = {'Very Strong': '#d62728', 'Strong': '#ff7f0e', 'Moderate': '#ffbb78', 
                             'Weak': '#2ca02c', 'Very Weak': '#98df8a'}
                    bar_colors = [colors.get(strength, '#1f77b4') for strength in strength_counts.index]
                    ax.bar(strength_counts.index, strength_counts.values, color=bar_colors)
                    ax.set_ylabel('Number of Sequences')
                    ax.set_title('Enhancer Strength Classification')
                    ax.tick_params(axis='x', rotation=45)
                    for i, v in enumerate(strength_counts.values):
                        ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Results table with enhanced information
                st.subheader("Detailed Results")
                
                # Display option for different classification schemes
                col1, col2, col3 = st.columns(3)
                with col1:
                    classification_scheme = st.selectbox(
                        "Classification scheme:",
                        ["Enhancer_Strength", "Binary_Prediction_999", "Binary_Prediction_99", "Binary_Prediction_95"]
                    )
                
                with col2:
                    show_only = st.selectbox(
                        "Filter results:",
                        ["All", "Very Strong only", "Strong+ only", "Moderate+ only"]
                    )
                
                with col3:
                    sort_by = st.selectbox(
                        "Sort by:",
                        ["Enhancer_Probability", "Sequence_ID", "Sequence_Length"],
                        index=0
                    )
                
                # Apply filters
                filtered_df = results_df.copy()
                if show_only == "Very Strong only":
                    filtered_df = filtered_df[filtered_df['Enhancer_Strength'] == 'Very Strong']
                elif show_only == "Strong+ only":
                    filtered_df = filtered_df[filtered_df['Enhancer_Probability'] > 0.99]
                elif show_only == "Moderate+ only":
                    filtered_df = filtered_df[filtered_df['Enhancer_Probability'] > 0.95]
                
                # Sort results
                ascending = True if sort_by != "Enhancer_Probability" else False
                filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
                
                # Select columns to display
                display_cols = ['Sequence_ID', 'Sequence_Length', 'Enhancer_Probability', 
                               classification_scheme, 'Enhancer_Strength']
                
                # Display table
                st.dataframe(
                    filtered_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button with enhanced results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="enhancer_predictions_detailed.csv",
                    mime="text/csv"
                )
                
                # Additional interpretation notes
                st.markdown("---")
                st.markdown("""
                ### 📋 Usage Notes:
                - **Relative ranking** is more meaningful than absolute scores
                - Sequences with probability differences < 0.01 should be considered similar
                - **Always validate** predictions experimentally
                - Consider this tool as a **first-pass screening** method only
                """)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please check that your sequences are in valid DNA format (A, T, G, C, N)")

if __name__ == "__main__":
    main()
