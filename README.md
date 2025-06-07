## scour-damage_classification

This repository provides code for classifying rail defects using vibration (acceleration) and speed data through a CNN that combines both inputs.

#### Scripts:

main.py – Loads, preprocesses, and analyzes sensor/speed data, trains the CNN model, and evaluates performance.

#### Datasets:
The MATLAB datasets contain vibration and speed measurements under different rail conditions:
Baseline: Healthy bridge (undamaged).
5% damage: Minor vertical stiffness reduction.
10% damage: Moderate vertical stiffness reduction.
20% damage: Moderate/High vertical stiffness reduction.

#### File Naming Convention:
Data04-08_{SensorPosition}_{Wagon}_Cut.mat – Vibration data (TF for front bogie and VG for car body position).
Data04-08_velocidade.mat – Corresponding speed measurements.

#### Key Features:

Data preprocessing: Normalization and balanced train/test splitting.
Hybrid CNN model: Processes both vibration signals and speed data.
Performance evaluation: Confusion matrices, accuracy boxplots, and loss curves.
Reproducibility: 20 independent runs for statistical reliability.
