# Improved CNN Model for Enhancer Detection
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.layers import Input, GlobalMaxPooling1D, Activation, concatenate, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import random
import os
import time

# Check for GPU
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# One-hot encoding function
def one_hot_encode(sequence):
    """Convert DNA sequence to one-hot encoding"""
    encoding = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0.25, 0.25, 0.25, 0.25]
    }
    return np.array([encoding.get(base.upper(), encoding['N']) for base in sequence])

# Load sequences from fasta files
def load_sequences_from_fasta(file_path):
    """Read DNA sequences from a FASTA file"""
    sequences = []
    ids = []
    try:
        with open(file_path, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequences.append(str(record.seq).upper())
                ids.append(record.id)
        print(f"Loaded {len(sequences)} sequences from {file_path}")
        return sequences, ids
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return [], []

# Data augmentation function with safeguards
def augment_data(sequences, labels, augmentation_factor=1, mutation_rate=0.02, gc_weighted=True):
    """
    Augment DNA sequences with point mutations.
    
    Args:
        sequences: List of DNA sequences
        labels: List of labels (0 or 1)
        augmentation_factor: How many augmented versions per original sequence
        mutation_rate: Probability of each base being mutated
        gc_weighted: Whether to weight mutations to maintain similar GC content
        
    Returns:
        Augmented sequences and labels
    """
    augmented_sequences = []
    augmented_labels = []
    
    # Keep originals
    augmented_sequences.extend(sequences)
    augmented_labels.extend(labels)
    
    # Only augment enhancers (positive class)
    enhancer_indices = [i for i, label in enumerate(labels) if label == 1]
    
    # Define mutation process
    def mutate_sequence(sequence):
        """Apply random mutations to a sequence"""
        mutated_sequence = list(sequence)
        for i in range(len(mutated_sequence)):
            if random.random() < mutation_rate:
                original = mutated_sequence[i].upper()
                
                if gc_weighted:
                    # Maintain similar GC content
                    if original in 'GC':
                        replacement = random.choice('GC')
                    else:
                        replacement = random.choice('AT')
                else:
                    # Completely random replacement
                    choices = [b for b in 'ACGT' if b != original]
                    replacement = random.choice(choices)
                    
                mutated_sequence[i] = replacement
                
        return ''.join(mutated_sequence)
    
    # Create augmented sequences
    print(f"Augmenting {len(enhancer_indices)} enhancer sequences...")
    for idx in enhancer_indices:
        seq = sequences[idx]
        label = labels[idx]
        
        for _ in range(augmentation_factor):
            augmented_sequences.append(mutate_sequence(seq))
            augmented_labels.append(label)
    
    print(f"Created {len(augmented_sequences)} sequences after augmentation")
    return augmented_sequences, augmented_labels

# Generate synthetic negative examples
def generate_negative_controls(count=1000, min_length=200, max_length=2000):
    """Generate synthetic negative controls"""
    negatives = []
    
    # Generate 4 types of negative controls
    types = ["poly_A", "poly_T", "random", "repeat"]
    
    for i in range(count):
        length = random.randint(min_length, max_length)
        type_idx = i % len(types)
        
        if types[type_idx] == "poly_A":
            # Poly-A sequence
            seq = "A" * length
            
        elif types[type_idx] == "poly_T":
            # Poly-T sequence
            seq = "T" * length
            
        elif types[type_idx] == "random":
            # Random sequence
            seq = ''.join(random.choice("ACGT") for _ in range(length))
            
        elif types[type_idx] == "repeat":
            # Repetitive pattern
            motif = ''.join(random.choice("ACGT") for _ in range(random.randint(3, 8)))
            repeats = length // len(motif) + 1
            seq = (motif * repeats)[:length]
        
        negatives.append(seq)
    
    return negatives

# Prepare data
def prepare_data(enhancer_file, negative_file, synthetic_negatives=1000, val_size=0.2, test_size=0.1):
    """
    Prepare training, validation and test datasets with proper class balance
    
    Args:
        enhancer_file: Path to FASTA file with enhancers
        negative_file: Path to FASTA file with negative examples
        synthetic_negatives: Number of synthetic negatives to generate
        val_size: Fraction of data for validation
        test_size: Fraction of data for final testing
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, max_length
    """
    # Load enhancers
    enhancers, enhancer_ids = load_sequences_from_fasta(enhancer_file)
    
    # Load or generate negatives
    if os.path.exists(negative_file):
        negatives, negative_ids = load_sequences_from_fasta(negative_file)
    else:
        print(f"Negative file {negative_file} not found. Generating synthetic negatives.")
        negatives = generate_negative_controls(synthetic_negatives)
        negative_ids = [f"synthetic_negative_{i}" for i in range(len(negatives))]
    
    # Balance classes if needed
    min_count = min(len(enhancers), len(negatives))
    min_count = min(min_count, 10000)  # Cap at 10k per class for efficiency
    
    print(f"Balancing classes to {min_count} samples each")
    if len(enhancers) > min_count:
        indices = random.sample(range(len(enhancers)), min_count)
        enhancers = [enhancers[i] for i in indices]
        enhancer_ids = [enhancer_ids[i] for i in indices]
    
    if len(negatives) > min_count:
        indices = random.sample(range(len(negatives)), min_count)
        negatives = [negatives[i] for i in indices]
        negative_ids = [negative_ids[i] for i in indices]
    
    # Combine data
    sequences = enhancers + negatives
    labels = [1] * len(enhancers) + [0] * len(negatives)
    sequence_ids = enhancer_ids + negative_ids
    
    # Determine maximum sequence length
    max_length = max(len(seq) for seq in sequences)
    max_length = min(max_length, 2000)  # Cap at 2000 bp to avoid memory issues
    print(f"Maximum sequence length: {max_length}")
    
    # Split into train, validation, and test
    # First split off test set
    train_val_seqs, test_seqs, train_val_labels, test_labels, train_val_ids, test_ids = train_test_split(
        sequences, labels, sequence_ids, test_size=test_size, stratify=labels, random_state=42
    )
    
    # Then split train into train and validation
    train_seqs, val_seqs, train_labels, val_labels, train_ids, val_ids = train_test_split(
        train_val_seqs, train_val_labels, train_val_ids, 
        test_size=val_size/(1-test_size), stratify=train_val_labels, random_state=42
    )
    
    print(f"Training set: {len(train_seqs)} sequences")
    print(f"Validation set: {len(val_seqs)} sequences")
    print(f"Test set: {len(test_seqs)} sequences")
    
    # Check class balance
    print(f"Training class balance: {sum(train_labels)}/{len(train_labels)} positive")
    print(f"Validation class balance: {sum(val_labels)}/{len(val_labels)} positive")
    print(f"Test class balance: {sum(test_labels)}/{len(test_labels)} positive")
    
    return (train_seqs, train_labels, train_ids,
            val_seqs, val_labels, val_ids,
            test_seqs, test_labels, test_ids,
            max_length)

# Process and encode sequences for model input
def process_sequences(sequences, labels, max_length, augment=False):
    """Process sequences into model-ready format"""
    # Augment if requested
    if augment:
        sequences, labels = augment_data(sequences, labels, augmentation_factor=1)
    
    # One-hot encode and pad
    X = []
    for seq in sequences:
        encoded = one_hot_encode(seq)
        X.append(encoded)
    
    X_padded = pad_sequences(X, maxlen=max_length, padding='post')
    y = np.array(labels)
    
    return X_padded, y

# Build an improved CNN model with residual connections
def build_model(input_shape, dropout_rate=0.5):
    """
    Build an improved CNN model for enhancer detection
    
    Args:
        input_shape: Shape of input sequences (length, 4)
        dropout_rate: Dropout probability
        
    Returns:
        Compiled model
    """
    # Input layer
    input_seq = Input(shape=input_shape, name='input')
    
    # First convolutional block
    conv1 = Conv1D(64, 7, activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_seq)
    pool1 = MaxPooling1D(2)(conv1)
    norm1 = BatchNormalization()(pool1)
    drop1 = Dropout(dropout_rate/2)(norm1)
    
    # Second convolutional block with residual connection
    conv2 = Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=l2(0.001))(drop1)
    # Ensure compatible shapes before adding
    res_conv = Conv1D(128, 1, padding='same')(drop1)  # 1x1 conv to match channels
    add2 = add([conv2, res_conv])  # Residual connection
    pool2 = MaxPooling1D(2)(add2)
    norm2 = BatchNormalization()(pool2)
    drop2 = Dropout(dropout_rate/2)(norm2)
    
    # Third convolutional block
    conv3 = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(drop2)
    pool3 = MaxPooling1D(2)(conv3)
    norm3 = BatchNormalization()(pool3)
    drop3 = Dropout(dropout_rate/2)(norm3)
    
    # Global pooling
    global_pool = GlobalMaxPooling1D()(drop3)
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(global_pool)
    norm4 = BatchNormalization()(dense1)
    drop4 = Dropout(dropout_rate)(norm4)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(drop4)
    
    # Create and compile model
    model = Model(inputs=input_seq, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# Train the model with cross-validation
def train_model_with_cv(train_seqs, train_labels, val_seqs, val_labels, max_length, 
                     n_splits=5, epochs=50, batch_size=32):
    """
    Train the model with cross-validation
    
    Args:
        train_seqs: Training sequences
        train_labels: Training labels
        val_seqs: Validation sequences
        val_labels: Validation labels
        max_length: Maximum sequence length
        n_splits: Number of CV folds
        epochs: Training epochs
        batch_size: Batch size
        
    Returns:
        Best model and training history
    """
    # Process and encode sequences
    X_train, y_train = process_sequences(train_seqs, train_labels, max_length, augment=True)
    X_val, y_val = process_sequences(val_seqs, val_labels, max_length)
    
    # Create output directory
    os.makedirs("model_output", exist_ok=True)
    
    # Initialize cross-validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []
    cv_scores = []
    best_val_acc = 0
    best_model = None
    
    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f"\nTraining fold {fold+1}/{n_splits}")
        
        # Get fold data
        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
        y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]
        
        # Build model
        model = build_model(input_shape=(max_length, 4))
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(f'model_output/fold_{fold+1}_best.h5', 
                            monitor='val_accuracy', save_best_only=True, mode='max')
        ]
        
        # Train model
        history = model.fit(
            X_fold_train, y_fold_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_fold_test, y_fold_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on this fold's test data
        scores = model.evaluate(X_fold_test, y_fold_test, verbose=0)
        print(f"Fold {fold+1} - Accuracy: {scores[1]:.4f}, Precision: {scores[2]:.4f}, Recall: {scores[3]:.4f}")
        
        # Save history for this fold
        histories.append(history.history)
        cv_scores.append(scores[1])  # accuracy
        
        # Check if this is the best model so far on the validation set
        val_scores = model.evaluate(X_val, y_val, verbose=0)
        val_acc = val_scores[1]
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            print(f"New best model found (validation accuracy: {val_acc:.4f})")
    
    # Save best model
    if best_model is not None:
        best_model.save('model_output/best_model.h5')
        print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")
    
    # Print average CV results
    print(f"\nCross-validation results:")
    print(f"Average accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    return best_model, histories

# Evaluate model on test set
def evaluate_model(model, test_seqs, test_labels, test_ids, max_length):
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained model
        test_seqs: Test sequences
        test_labels: Test labels
        test_ids: Test sequence IDs
        max_length: Maximum sequence length
    """
    # Process test data
    X_test, y_test = process_sequences(test_seqs, test_labels, max_length)
    
    # Evaluate on test set
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("\nTest set evaluation:")
    print(f"Loss: {scores[0]:.4f}")
    print(f"Accuracy: {scores[1]:.4f}")
    print(f"Precision: {scores[2]:.4f}")
    print(f"Recall: {scores[3]:.4f}")
    print(f"AUC: {scores[4]:.4f}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(np.int32).flatten()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    
    # Create detailed results dataframe
    import pandas as pd
    results_df = pd.DataFrame({
        'Sequence_ID': test_ids,
        'True_Label': y_test,
        'Predicted_Probability': y_pred.flatten(),
        'Predicted_Label': y_pred_binary
    })
    
    # Save results
    results_df.to_csv('model_output/test_results.csv', index=False)
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('model_output/precision_recall_curve.png')
    
    # Plot histogram of prediction probabilities by class
    plt.figure(figsize=(12, 6))
    plt.hist(y_pred[y_test==0], bins=20, alpha=0.5, label='Non-enhancers')
    plt.hist(y_pred[y_test==1], bins=20, alpha=0.5, label='Enhancers')
    plt.xlabel('Enhancer Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.savefig('model_output/prediction_distribution.png')
    
    # Plot ROC Curve
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('model_output/roc_curve.png')
    
    return results_df

# Test with specific control sequences
def test_with_controls(model, max_length):
    """Test model with specific control sequences"""
    print("\nTesting model with control sequences:")
    
    # Define control sequences
    controls = [
        ("Strong enhancer", "CACGTGGCCCAGCCTGCTGCTGTGGGCCGCACGTGGCGCACGTGGGCCAATCAGCAGGTGTTAATGCAGATAAACCCACCACGTGGTGGGAACACACGTGGAGATAGATATAAAGGAAGGAATGTTCT"),
        ("Poly-A", "A" * 500),
        ("Poly-T", "T" * 500),
        ("Random", ''.join(random.choice('ACGT') for _ in range(500))),
        ("Promoter-like", "TATAAAAGGCGCGATTGCTATAATCACGCAGCGGTGAGCGTAGCGTCACTCACGCAACGCACGCGACAGCACGCAGCTCAGCTCCTCGCTCATTGGTACGCTCGCTCGCTCGCTCGCTCGCCTAGCTAGCTAGCTAGT"),
        ("Insulator", "CTCFCTCFCTCGCGGCCGGCAGAGGAGCAGCCCCTGCCGGCAGCATGGCGGCAGCGACTCCAAGCGCTTTGTGGGTGCGCGAGCGCGCGCGCAGGGGCGGACGCGCCGCGTCCGCCCCGCCTCCCCCGCCCCCGCCCCGCTCCTGGTAGCGGCCGCGCAGCGACAGCGCCGCCTCGTCGCCACCGCTTCCCGCCCCGCCCCCGCGCCGCCTTTGAAAGGCGGCAGCGCGCGCTCCCGCGGCGCGGTCCCAGCCTCGTCTCCCCGCCCCCTCCCTCCCCTCCCTCCCCTTCTCCTCCCTCGCTCGCTCGCTCGCTCCCCGCCCCCTGCCCCTCCACCCGCCCCCTCTCC")
    ]
    
    # Process and predict
    for name, seq in controls:
        encoded = one_hot_encode(seq)
        padded = pad_sequences([encoded], maxlen=max_length, padding='post')
        pred = model.predict(padded, verbose=0)[0][0]
        
        print(f"{name}: {pred:.4f} → {'Enhancer' if pred > 0.5 else 'Non-enhancer'}")
    
    print("Control sequence testing complete")

# Helper function to plot training history
def plot_training_history(histories):
    """Plot training metrics"""
    plt.figure(figsize=(15, 10))
    
    # Calculate average metrics across folds
    avg_acc = np.mean([h['accuracy'] for h in histories], axis=0)
    avg_val_acc = np.mean([h['val_accuracy'] for h in histories], axis=0)
    avg_loss = np.mean([h['loss'] for h in histories], axis=0)
    avg_val_loss = np.mean([h['val_loss'] for h in histories], axis=0)
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(avg_acc, label='Training')
    plt.plot(avg_val_acc, label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(avg_loss, label='Training')
    plt.plot(avg_val_loss, label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot precision and recall if available
    if 'precision' in histories[0]:
        avg_precision = np.mean([h['precision'] for h in histories], axis=0)
        avg_val_precision = np.mean([h['val_precision'] for h in histories], axis=0)
        avg_recall = np.mean([h['recall'] for h in histories], axis=0)
        avg_val_recall = np.mean([h['val_recall'] for h in histories], axis=0)
        
        plt.subplot(2, 2, 3)
        plt.plot(avg_precision, label='Training')
        plt.plot(avg_val_precision, label='Validation')
        plt.title('Model Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(avg_recall, label='Training')
        plt.plot(avg_val_recall, label='Validation')
        plt.title('Model Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_output/training_history.png')
    plt.show()

# Main function to run the entire workflow
def main():
    """Run the complete enhancer detection workflow"""
    start_time = time.time()
    
    # Define file paths
    enhancer_file = "enhancers.fasta"
    negative_file = "negative_controls.fasta"
    synthetic_negatives = 10000  # How many synthetic negatives to generate if file not found
    
    # Prepare data
    train_seqs, train_labels, train_ids, val_seqs, val_labels, val_ids, test_seqs, test_labels, test_ids, max_length = prepare_data(
        enhancer_file, negative_file, synthetic_negatives
    )
    
    # Train model
    model, histories = train_model_with_cv(
        train_seqs, train_labels, val_seqs, val_labels, max_length,
        n_splits=5, epochs=50, batch_size=32
    )
    
    # Plot training history
    plot_training_history(histories)
    
    # Evaluate model
    results_df = evaluate_model(model, test_seqs, test_labels, test_ids, max_length)
    
    # Test with control sequences
    test_with_controls(model, max_length)
    
    # Print execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return model, results_df

# Run the main function if executed directly
if __name__ == "__main__":
    main()
