# Import necessary libraries
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalMaxPooling1D, Masking, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from kerastuner.tuners import RandomSearch
import random
import matplotlib.pyplot as plt

# One-hot encoding function
def one_hot_encode(sequence):
    encoding = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0.25, 0.25, 0.25, 0.25]
    }
    return np.array([encoding.get(base.upper(), [0.25, 0.25, 0.25, 0.25]) for base in sequence], dtype=np.float32)

# Load sequences from fasta files
def load_sequences_from_fasta(file_path):
    with open(file_path, "r") as handle:
        sequences = [str(record.seq).upper() for record in SeqIO.parse(handle, "fasta")]
    return sequences

# Data augmentation function
def mutate_sequence(sequence, mutation_rate=0.01):
    nucleotides = ['A', 'C', 'G', 'T']
    mutated_sequence = list(sequence)
    for i in range(len(mutated_sequence)):
        if random.random() < mutation_rate:
            mutated_sequence[i] = random.choice(nucleotides)
    return ''.join(mutated_sequence)

def augment_data(sequences, labels, augmentation_factor=1, mutation_rate=0.01):
    augmented_sequences = []
    augmented_labels = []
    for seq, label in zip(sequences, labels):
        augmented_sequences.append(seq)
        augmented_labels.append(label)
        for _ in range(augmentation_factor):
            augmented_sequences.append(mutate_sequence(seq, mutation_rate))
            augmented_labels.append(label)
    return augmented_sequences, augmented_labels

# Load sequences
experimentally_derived_file = "/content/drive/MyDrive/extracted_sequences.fasta"
shadow_derived_file = "/content/drive/MyDrive/shadow/shadow_new.fasta"
negative_file = "/content/drive/MyDrive/neg.fa"

experimental_sequences = load_sequences_from_fasta(experimentally_derived_file)
shadow_sequences = load_sequences_from_fasta(shadow_derived_file)
negative_sequences = load_sequences_from_fasta(negative_file)

# Split sequences into training and validation sets
train_experimental, val_experimental = train_test_split(experimental_sequences, test_size=0.2, random_state=42)
train_shadow, val_shadow = train_test_split(shadow_sequences, test_size=0.2, random_state=42)

# Combine the sequences and labels
train_data = train_experimental + train_shadow + negative_sequences
train_labels = [1]*len(train_experimental) + [1]*len(train_shadow) + [0]*len(negative_sequences)
val_data = val_experimental + val_shadow
val_labels = [1]*len(val_experimental) + [1]*len(val_shadow)

# Augment the training data
augmented_train_data, augmented_train_labels = augment_data(train_data, train_labels, augmentation_factor=2, mutation_rate=0.01)

# Define the maximum sequence length
max_length = max(max(map(len, augmented_train_data)), max(map(len, val_data)))

# Data Generator
def data_generator(sequences, labels, batch_size, max_length):
    while True:
        index = 0
        while index < len(sequences):
            batch_seqs = sequences[index: index + batch_size]
            batch_labels = labels[index: index + batch_size]
            encoded_batch = [one_hot_encode(seq) for seq in batch_seqs]
            padded_batch = pad_sequences(encoded_batch, maxlen=max_length, padding='post')
            yield np.array(padded_batch), np.array(batch_labels)
            index += batch_size

batch_size = 8
train_gen = data_generator(augmented_train_data, augmented_train_labels, batch_size, max_length)
val_gen = data_generator(val_data, val_labels, batch_size, max_length)
steps_per_epoch = len(augmented_train_data) // batch_size
validation_steps = len(val_data) // batch_size

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', np.unique(augmented_train_labels), augmented_train_labels)
class_weights_dict = dict(enumerate(class_weights))

# Build baseline model with improvements
def build_baseline_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the baseline model with class weights and callbacks
baseline_model = build_baseline_model((max_length, 4))
baseline_model.fit(train_gen, epochs=20, validation_data=val_gen, class_weight=class_weights_dict, callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True)
scores = []

# Convert sequences to one-hot encoded format
X = np.array([one_hot_encode(seq) for seq in augmented_train_data])
X = pad_sequences(X, maxlen=max_length, padding='post')
y = np.array(augmented_train_labels)

for train, test in kfold.split(X, y):
    model = build_baseline_model((max_length, 4))
    model.fit(X[train], y[train], epochs=20, class_weight=class_weights_dict)
    score = model.evaluate(X[test], y[test])
    scores.append(score[1])

print("Average accuracy: ", np.mean(scores))

# Function to build the model with tunable hyperparameters
def build_tuned_model(hp):
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value