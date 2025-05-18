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
    except:
        st.error("Model file 'best_model.h5' not found. Please ensure the model is in the app directory.")
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

# Main app
def main():
    st.title("🧬 DNA Enhancer Detection Tool")
    st.markdown("---")
    
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
        4. Download results as CSV
        """
    )
    
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
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Sequence_ID': sequence_ids,
                    'Sequence_Length': [len(seq) for seq in sequences],
                    'Enhancer_Probability': predictions,
                    'Prediction': ['Enhancer' if p > 0.5 else 'Non-enhancer' for p in predictions],
                    'Confidence': ['High' if abs(p - 0.5) > 0.3 else 'Medium' if abs(p - 0.5) > 0.1 else 'Low' for p in predictions]
                })
                
                # Display results
                st.header("📊 Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total sequences", len(results_df))
                with col2:
                    predicted_enhancers = (results_df['Enhancer_Probability'] > 0.5).sum()
                    st.metric("Predicted enhancers", predicted_enhancers)
                with col3:
                    st.metric("Average probability", f"{results_df['Enhancer_Probability'].mean():.3f}")
                with col4:
                    high_confidence = (results_df['Confidence'] == 'High').sum()
                    st.metric("High confidence predictions", high_confidence)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Probability distribution
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(results_df['Enhancer_Probability'], bins=20, alpha=0.7, edgecolor='black')
                    ax.axvline(x=0.5, color='red', linestyle='--', label='Decision threshold')
                    ax.set_xlabel('Enhancer Probability')
                    ax.set_ylabel('Number of Sequences')
                    ax.set_title('Distribution of Enhancer Probabilities')
                    ax.legend()
                    st.pyplot(fig)
                
                with col2:
                    # Prediction counts
                    fig, ax = plt.subplots(figsize=(8, 6))
                    prediction_counts = results_df['Prediction'].value_counts()
                    colors = ['#ff7f0e' if pred == 'Enhancer' else '#1f77b4' for pred in prediction_counts.index]
                    ax.bar(prediction_counts.index, prediction_counts.values, color=colors)
                    ax.set_ylabel('Number of Sequences')
                    ax.set_title('Prediction Summary')
                    for i, v in enumerate(prediction_counts.values):
                        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
                    st.pyplot(fig)
                
                # Results table
                st.subheader("Detailed Results")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    show_only = st.selectbox(
                        "Filter results:",
                        ["All", "Enhancers only", "Non-enhancers only", "High confidence only"]
                    )
                
                with col2:
                    sort_by = st.selectbox(
                        "Sort by:",
                        ["Enhancer_Probability", "Sequence_ID", "Sequence_Length"],
                        index=0
                    )
                
                # Apply filters
                filtered_df = results_df.copy()
                if show_only == "Enhancers only":
                    filtered_df = filtered_df[filtered_df['Prediction'] == 'Enhancer']
                elif show_only == "Non-enhancers only":
                    filtered_df = filtered_df[filtered_df['Prediction'] == 'Non-enhancer']
                elif show_only == "High confidence only":
                    filtered_df = filtered_df[filtered_df['Confidence'] == 'High']
                
                # Sort results
                ascending = True if sort_by != "Enhancer_Probability" else False
                filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
                
                # Display table
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="enhancer_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please check that your sequences are in valid DNA format (A, T, G, C, N)")

if __name__ == "__main__":
    main()
