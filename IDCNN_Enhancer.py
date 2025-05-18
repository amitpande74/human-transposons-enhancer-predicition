# Import necessary libraries
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from kerastuner import HyperModel
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
experimentally_derived_file = "path_to_experimental_sequences.fasta"
negative_file = "path_to_negative_sequences.fasta"

experimental_sequences = load_sequences_from_fasta(experimentally_derived_file)
negative_sequences = load_sequences_from_fasta(negative_file)

# Split sequences into training and validation sets
train_experimental, val_experimental = train_test_split(experimental_sequences, test_size=0.2, random_state=42)
train_negative, val_negative = train_test_split(negative_sequences, test_size=0.2, random_state=42)

# Combine the sequences and labels
train_data = train_experimental + train_negative
train_labels = [1]*len(train_experimental) + [0]*len(train_negative)
val_data = val_experimental + val_negative
val_labels = [1]*len(val_experimental) + [0]*len(val_negative)

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

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', np.unique(augmented_train_labels), augmented_train_labels)
class_weights_dict = dict(enumerate(class_weights))

# CNNHyperModel Class Definition
class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(Conv1D(filters=hp.Int('filters', min_value=32, max_value=256, step=32),
                         kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling1D(2))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))
        model.add(Flatten())
        model.add(Dense(units=hp.Int('units', min_value=32, max_value=256, step=32), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

# Initialize and Run the Tuner
hypermodel = CNNHyperModel(input_shape=(max_length, 4))

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='hparam_tuning'
)

tuner.search(train_gen, 
             epochs=20, 
             validation_data=val_gen, 
             callbacks=[EarlyStopping(monitor='val_loss', patience=10), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5), ModelCheckpoint('best_model.h5', save_best_only=True)])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Convert sequences to one-hot encoded format
X = np.array([one_hot_encode(seq) for seq in augmented_train_data])
X = pad_sequences(X, maxlen=max_length, padding='post')
y = np.array(augmented_train_labels)

# Cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train, test in kfold.split(X, y):

    # Create a new instance of the best model
    model = tf.keras.models.clone_model(best_model)

    # Compile the cloned model (use the same optimizer and loss as the best model)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X[train], y[train], epochs=20, batch_size=batch_size, class_weight=class_weights_dict)

    # Evaluate the model
    scores = model.evaluate(X[test], y[test], verbose=0)
    print(f'Score for fold {len(cv_scores)+1}: {model.metrics_names[1]} of {scores[1]}')
    cv_scores.append(scores[1])

# Average score from all k-folds
print(f'Average accuracy from cross-validation: {np.mean(cv_scores)}')
