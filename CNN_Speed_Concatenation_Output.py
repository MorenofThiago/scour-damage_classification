# -*- coding: utf-8 -*-
"""
CNN model for scour damage classification with speed data integration
Author: Thiago Moreno Fernandes
Federal University of Santa Catarina - Brazil
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Configuration
plt.rcParams['font.family'] = 'Times New Roman'
start_time = time.time()

# Parameters
N_RUNS = 20
TRAIN_SAMPLES = 50
TEST_SAMPLES = 50
SENSOR_POSITION = 'TF'  # 'TF' or 'VG'
WAGON = 'PrimVag'

def load_and_preprocess_data():
    """Load and normalize sensor and speed data"""
    # Load sensor data
    data = loadmat(f'Data04-08_{SENSOR_POSITION}_{WAGON}_Cut.mat')
    sensor_data = {
        'baseline': data['Baseline'],
        '5%': data['CincoP'],
        '10%': data['DezP'],
        '20%': data['VinteP']
    }
    
    # Load speed data
    speed = loadmat('Data04-08_velocidade.mat')
    speed_data = {
        'baseline': speed['veloc_baseline'],
        '5%': speed['veloc_cincoP'],
        '10%': speed['veloc_dezP'],
        '20%': speed['veloc_vinteP']
    }
    
    # Normalize all data
    def normalize(x): return (x - np.min(x)) / (np.ptp(x))
    
    sensor_data = {k: normalize(v) for k, v in sensor_data.items()}
    speed_data = {k: normalize(v) for k, v in speed_data.items()}
    
    return sensor_data, speed_data

def create_dataframe(sensor_data, speed_data):
    """Create labeled dataframe with sensor and speed data"""
    dfs = []
    for i, (scenario, data) in enumerate(sensor_data.items()):
        df = pd.DataFrame(data)
        df['label'] = i
        df['speed'] = speed_data[scenario].flatten()
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df.sample(frac=1, random_state=42).reset_index(drop=True)

def prepare_datasets(df):
    """Prepare train/test datasets"""
    train = df.sample(n=TRAIN_SAMPLES*4, random_state=42)
    test = df.drop(train.index)
    
    # Prepare test samples per scenario
    test_samples = pd.concat([
        test[test['label'] == i].sample(n=TEST_SAMPLES, random_state=42) 
        for i in range(4)
    ])
    
    # Format data
    def format_data(data):
        x = data.drop(['label', 'speed'], axis=1).values.reshape(-1, 5830, 1)
        y = tf.keras.utils.to_categorical(data['label'], num_classes=4)
        speed = np.array(data['speed'], dtype=float)
        return x, y, speed
    
    x_train, y_train, speed_train = format_data(train)
    x_test, y_test, speed_test = format_data(test_samples)
    
    return (x_train, y_train, speed_train), (x_test, y_test, speed_test)

def build_model():
    """Build CNN model with speed input"""
    initializer = tf.keras.initializers.GlorotNormal()
    
    # Input layers
    signal_input = tf.keras.Input(shape=(5830, 1), name='signal_input')
    speed_input = tf.keras.Input(shape=(1,), name='speed_input')
    
    if SENSOR_POSITION == 'TF':
        # TF position architecture
        x = tf.keras.layers.Conv1D(128, 5, activation='relu', kernel_initializer=initializer)(signal_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        
        x = tf.keras.layers.Conv1D(96, 4, activation='relu', kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        
        x = tf.keras.layers.Conv1D(32, 5, activation='relu', kernel_initializer=initializer)(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        
        x = tf.keras.layers.Conv1D(96, 3, activation='relu', kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        
        x = tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        
    else:  # VG position
        x = tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer)(signal_input)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer)(x)
        x = tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer)(x)
        x = tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer)(x)
        x = tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer)(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    # Process speed input
    speed_x = tf.keras.layers.Dense(tf.keras.backend.int_shape(x)[-1], 
                                   activation='relu', 
                                   kernel_initializer=initializer)(speed_input)
    
    # Combine features
    combined = tf.keras.layers.Concatenate()([x, speed_x])
    x = tf.keras.layers.Dense(32 if SENSOR_POSITION == 'TF' else 48, 
                             activation='relu', 
                             kernel_initializer=initializer)(combined)
    
    # Output
    output = tf.keras.layers.Dense(4, activation='softmax')(x)
    
    # Compile model
    model = tf.keras.Model(inputs=[signal_input, speed_input], outputs=output)
    lr = 0.000169 if SENSOR_POSITION == 'TF' else 0.00015
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy'])
    
    model.summary()
    return model

def train_and_evaluate(model, train_data, test_data, run_idx):
    """Train model and evaluate performance"""
    x_train, y_train, speed_train = train_data
    x_test, y_test, speed_test = test_data
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                         min_lr=1e-6, verbose=1)
    ]
    
    history = model.fit(
        [x_train, speed_train],
        y_train,
        epochs=400,
        batch_size=10,
        validation_split=0.2,
        verbose=1,
        callbacks=callbacks
    )
    
    # Plot training history
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.savefig(f'Loss_ConcOutput_{SENSOR_POSITION}_n{TRAIN_SAMPLES}_it{run_idx}.png', 
                dpi=600, bbox_inches='tight')
    plt.close()
    
    # Evaluate
    y_pred = model.predict([x_test, speed_test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    return (confusion_matrix(y_true_classes, y_pred_classes),
            accuracy_score(y_true_classes, y_pred_classes),
            y_true_classes, y_pred_classes)

def plot_results(best_conf_matrix, accuracies):
    """Plot confusion matrix and accuracy boxplot"""
    class_names = ['Baseline', '5%', '10%', '20%']
    
    # Normalized confusion matrix
    conf_norm = best_conf_matrix.astype('float') / best_conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_norm, annot=True, fmt='.2f', cmap='Blues', 
                annot_kws={"size": 20}, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=22)
    plt.ylabel('True', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(f'ConfusionMatrix_Speed_ConcOutput_{SENSOR_POSITION}_n{TRAIN_SAMPLES}.png', 
                dpi=600, bbox_inches='tight')
    plt.close()
    
    # Accuracy boxplot
    plt.figure(figsize=(8, 6))
    acc_df = pd.DataFrame(accuracies)
    
    box = plt.boxplot([acc_df[col] for col in acc_df.columns], patch_artist=True)
    colors = ['green', 'goldenrod', 'darkorange', 'darkred']
    
    for i, (patch, color) in enumerate(zip(box['boxes'], colors)):
        patch.set_edgecolor('black')
        patch.set_facecolor('none')
        patch.set_linewidth(1)
        
        for whisker in box['whiskers'][2*i:2*i+2]:
            whisker.set_color('black')
            whisker.set_linewidth(1)
            whisker.set_linestyle((0, (8, 6)))
            
        for cap in box['caps'][2*i:2*i+2]:
            cap.set_color('black')
            cap.set_linewidth(1)
            
        box['medians'][i].set_color(color)
        box['medians'][i].set_linewidth(2)
    
    plt.xlabel('Scenario', fontsize=24, fontfamily='serif')
    plt.ylabel('Accuracy', fontsize=24, fontfamily='serif')
    plt.xticks(ticks=range(1, len(acc_df.columns)+1, labels=class_names, 
               fontsize=22, fontfamily='serif')
    plt.yticks(fontsize=22, fontfamily='serif')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'Boxplot_Speed_ConcOutput_{SENSOR_POSITION}_n{TRAIN_SAMPLES}.png', 
                dpi=600, bbox_inches='tight')
    plt.close()

def main():
    """Main execution pipeline"""
    sensor_data, speed_data = load_and_preprocess_data()
    df = create_dataframe(sensor_data, speed_data)
    train_data, test_data = prepare_datasets(df)
    
    best_accuracy = 0
    best_conf_matrix = None
    accuracies = {scenario: [] for scenario in ['Baseline', '5%', '10%', '20%']}
    
    for run in range(N_RUNS):
        print(f"Run {run+1}/{N_RUNS}")
        model = build_model()
        conf_matrix, accuracy, y_true, y_pred = train_and_evaluate(
            model, train_data, test_data, run)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_conf_matrix = conf_matrix
        
        # Record scenario accuracies
        for i, scenario in enumerate(accuracies.keys()):
            mask = (y_true == i)
            accuracies[scenario].append(accuracy_score(y_true[mask], y_pred[mask]))
    
    plot_results(best_conf_matrix, accuracies)
    print(f"Total execution time: {(time.time()-start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()
