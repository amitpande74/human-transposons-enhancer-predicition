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
    page_title="DNA Enhancer Detection Tool",
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
    return np.array([encoding.get(base.upper(), encoding['N']) for base in sequence])

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

def classify_sequence_type(probability):
    """Classify the regulatory element type based on probability"""
    if probability >= 0.9:
        return "Strong Enhancer", "High"
    elif probability >= 0.7:
        return "Likely Enhancer", "Medium"
    elif probability >= 0.5:
        return "Possible Enhancer", "Low"
    elif probability <= 0.1:
        return "Likely Insulator/Non-regulatory", "High"
    elif probability <= 0.3:
        return "Likely Non-enhancer", "Medium"
    else:
        return "Uncertain/Promoter-like", "Low"

# Main app
def main():
    st.title("🧬 DNA Enhancer Detection Tool")
    st.markdown("---")
    
    # Add information about model capabilities
    st.info("""
    📌 **About This Tool**: This deep learning model has been trained to identify DNA enhancer
    sequences based on their sequence characteristics. It can distinguish enhancers from other
    regulatory elements like insulators and, to some extent, promoters.
    
    The model analyzes sequence features such as transcription factor binding motifs, GC content,
    and other DNA patterns associated with enhancer activity.
    """)
    
    # Sidebar
    st.sidebar.header("About This Tool")
    st.sidebar.info(
        """
        This tool uses a 1D CNN deep learning model to predict whether 
        DNA sequences are enhancers. Upload your sequences in FASTA format 
        or paste them directly to get predictions.
        """
    )
    
    st.sidebar.header("Model Performance")
    st.sidebar.info("""
    **Validation metrics:**
    - **Accuracy**: 86.5%
    - **Precision**: 83.8%
    - **Recall**: 98.7%
    
    **Element type detection:**
    - **Enhancers**: 98.6% accuracy
    - **Insulators**: 99.9% accuracy
    - **Promoters**: Mixed classification
    """)
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        """
        1. Choose input method (upload file or paste text)
        2. Provide DNA sequences in FASTA format
        3. Click 'Predict Enhancers' to get results
        4. Download results as CSV
        """
    )
    
    # Note about enhancer types
    st.sidebar.header("About Enhancer Types")
    st.sidebar.info("""
    **Note on enhancer types:**
    
    This model detects standard enhancers. For specialized enhancer types:
    
    - **Shadow enhancers**: Sets of redundant enhancers that regulate the same gene
    - **Super-enhancers**: Large clusters of enhancers with unusually high TF binding
    
    These specialized types require additional analysis beyond sequence features alone.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Set max_length based on your training
    MAX_LENGTH = 1000
    
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
    
    # Example sequences
    with st.expander("Need test sequences? Try these examples"):
        st.markdown("""
        **Copy and paste any of these example sequences:**
        
        1. **Known Enhancer:**
        ```
        >known_enhancer
        CACGTGGCCCAGCCTGCTGCTGTGGGCCGCACGTGGCGCACGTGGGCCAATCAGCAGGTGTTAATGCAGATAAACCCACCACGTGGTGGGAACACACGTGGAGATAGATATAAAGGAAGGAATGTTCT
        ```
        
        2. **Insulator (CTCF binding site):**
        ```
        >insulator_ctcf
        AGTCCCCTCTCGCGGCCGGCAGAGGAGCAGCCCCTGCCGGCAGCATGGCGGCAGCGACTCCAAGCGCTTTGTGGGTGCGCGAGCGCGCGCGCAGGGGCGGACGCGCCGCGTCCGCCCCGCCTCCCCCGCCCCCGCCCCGCTCCTGGTAGCGGCCGCGCAGCGACAGCGCCGCCTCGTCGCCACCGCTTCCCGCCCCGCCCCCGCGCCGCCTTTGAAAGGCGGCAGCGCGCGCTCCCGCGGCGCGGTCCCAGCCTCGTCTCCCCGCCCCCTCCCTCCCCTCCCTCCCCTTCTCCTCCCTCGCTCGCTCGCTCGCTCCCCGCCCCCTGCCCCTCCACCCGCCCCCTCTCCACGCCACCCCCGCCCTC
        ```
        
        3. **Promoter:**
        ```
        >promoter_example
        TATAAAAGGCGCGATTGCTATAATCACGCAGCGGTGAGCGTAGCGTCACTCACGCAACGCACGCGACAGCACGCAGCTCAGCTCCTCGCTCATTGGTACGCTCGCTCGCTCGCTCGCTCGCCTAGCTAGCTAGCTAGT
        ```
        """)
    
    # Prediction
    if sequences and st.button("🔬 Predict Enhancers", type="primary"):
        with st.spinner("Analyzing sequences..."):
            try:
                # Make predictions
                predictions = predict_enhancers(model, sequences, MAX_LENGTH)
                
                # Create results dataframe with additional type classification
                types_confidences = [classify_sequence_type(p) for p in predictions]
                types = [t[0] for t in types_confidences]
                confidences = [t[1] for t in types_confidences]
                
                results_df = pd.DataFrame({
                    'Sequence_ID': sequence_ids,
                    'Sequence_Length': [len(seq) for seq in sequences],
                    'Enhancer_Probability': predictions,
                    'Element_Type': types,
                    'Confidence': confidences
                })
                
                # Display results
                st.header("📊 Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total sequences", len(results_df))
                with col2:
                    enhancer_count = sum(1 for p in predictions if p >= 0.5)
                    st.metric("Predicted enhancers", f"{enhancer_count} ({enhancer_count/len(predictions)*100:.1f}%)")
                with col3:
                    high_confidence = sum(1 for c in confidences if c == "High")
                    st.metric("High confidence predictions", high_confidence)
                with col4:
                    avg_prob = np.mean(predictions)
                    st.metric("Average probability", f"{avg_prob:.3f}")
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Probability distribution
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(results_df['Enhancer_Probability'], bins=20, alpha=0.7, edgecolor='black')
                    ax.axvline(x=0.5, color='r', linestyle='--', label='Decision threshold')
                    ax.axvline(x=0.9, color='g', linestyle='--', label='High confidence')
                    ax.axvline(x=0.1, color='b', linestyle='--', label='High confidence (non-enhancer)')
                    ax.set_xlabel('Enhancer Probability')
                    ax.set_ylabel('Number of Sequences')
                    ax.set_title('Distribution of Enhancer Probabilities')
                    ax.legend()
                    st.pyplot(fig)
                
                with col2:
                    # Element type distribution
                    fig, ax = plt.subplots(figsize=(8, 6))
                    type_counts = results_df['Element_Type'].value_counts()
                    colors = {'Strong Enhancer': '#ff7f0e', 
                              'Likely Enhancer': '#ffbb78', 
                              'Possible Enhancer': '#ffd8b1',
                              'Uncertain/Promoter-like': '#c7c7c7', 
                              'Likely Non-enhancer': '#98df8a', 
                              'Likely Insulator/Non-regulatory': '#2ca02c'}
                    type_colors = [colors.get(t, '#1f77b4') for t in type_counts.index]
                    ax.bar(type_counts.index, type_counts.values, color=type_colors)
                    ax.set_ylabel('Number of Sequences')
                    ax.set_title('Element Type Distribution')
                    plt.xticks(rotation=45, ha='right')
                    for i, v in enumerate(type_counts.values):
                        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Results table
                st.subheader("Detailed Results")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    show_only = st.selectbox(
                        "Filter results:",
                        ["All", "Enhancers only", "Non-enhancers only", "Strong enhancers only", "Uncertain/Promoter-like"]
                    )
                
                with col2:
                    sort_by = st.selectbox(
                        "Sort by:",
                        ["Enhancer_Probability", "Sequence_ID", "Sequence_Length", "Element_Type"],
                        index=0
                    )
                
                # Apply filters
                filtered_df = results_df.copy()
                if show_only == "Enhancers only":
                    filtered_df = filtered_df[filtered_df['Enhancer_Probability'] >= 0.5]
                elif show_only == "Non-enhancers only":
                    filtered_df = filtered_df[filtered_df['Enhancer_Probability'] < 0.5]
                elif show_only == "Strong enhancers only":
                    filtered_df = filtered_df[filtered_df['Enhancer_Probability'] >= 0.9]
                elif show_only == "Uncertain/Promoter-like":
                    filtered_df = filtered_df[(filtered_df['Enhancer_Probability'] >= 0.3) & 
                                              (filtered_df['Enhancer_Probability'] < 0.7)]
                
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
                
                # Interpretation guidelines
                st.markdown("---")
                st.subheader("📋 How to Interpret Results")
                st.markdown("""
                - **Strong Enhancer** (≥0.9): Sequences with strong enhancer characteristics
                - **Likely Enhancer** (0.7-0.9): Sequences that have most enhancer features  
                - **Possible Enhancer** (0.5-0.7): Sequences with some enhancer-like features
                - **Uncertain/Promoter-like** (0.3-0.5): May be promoters or weak enhancers
                - **Likely Non-enhancer** (0.1-0.3): Probably not enhancers
                - **Likely Insulator/Non-regulatory** (≤0.1): Strong non-enhancer signature
                
                **Note:** Validation testing shows the model is highly accurate at distinguishing enhancers from insulators 
                (99.9% accuracy), but promoters show mixed classification due to their functional similarity to enhancers.
                """)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please check that your sequences are in valid DNA format (A, T, G, C, N)")

if __name__ == "__main__":
    main()
