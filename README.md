# Investigation-of-different-activationfunctions
The goal of this experiment was to investigate the impact of different activation functions on the prediction quality of a neural network in a regression task.

# Motivation & Expaction
While learning 'machine learning and AI', I noticed that many books and tutorials describe activation funtions theoretically,  but practical comparisons are often missing.

Activation functions are a fundamental part of neural networks. The theory behind it, can be found in several sources by own researches.
My expectation is that with the same test setup, question, and conditions, I'll get different results with different activation functions. 

This repository documents a hands-on experiment comparing different activation functions under identical training conditions.

# Test environment
## Hardware
  * CPU: Intel Core i3-12100F (4 cores / 8 threads)
  * GPU: Nvidia RTX 3050 - 6GB VRAM
  * RAM: 16 GB (dual-channel)
## Software
  * OS: Ubuntu 22.04.05 LTS (64bit)
  * Visual Studio Code
  * Python 3.11.15
  * TensorFlow incl. CUDA
  * Scikit-Learn
  * Pandas

# Dataset
The Iris-dataset contains measurements of iris flowers. Ingested via .csv-file.
## Features:
  * sepal length
  * sepal width
  * petal length
  * petal width
## Taget:
  * sepal length
  * type of problem: regression

# Data preparation
The following preprocessing steps were applied:
1. Due to low correlation: removing feature sepal.width
2. One-Hot-Encoding (OHE) of the species column
3. Standard scaling of:
   * petal.length
   * petal.width
4. Train-test split:
   * 80% training
   * 20% testing
  
# Architectures
Following section describes the used architectures of this experiment.
## Modelarchitecture
The model architecture stays fixed for all upcoming activation-function tests:
```text
  [ Input ] ──> ( 4 Features )
       │
       ▼
  [ Layer ] ──> Dense (32, Activation)
       │
       ▼
  [ Layer ] ──> Dense (64, Activation)
       │
       ▼
  [ Output ] ──> Dense (1)
```
## Networkarchitecture
The network architecture stays fixed for all upcoming activation-function tests.
  * Dense: 32 - 64 - 1
  * Optimizer: Adam
  * Loss: Mean Absolute Error (MAE)
  * Epochs: 70
  * Batch size: 32

# Experimental Design
Only the activation function was changed - all other parameters remained identical.
Tested activation functions:
  * ReLU
  * Leaky ReLU
  * ELU
  * SeLU
  * GELU
  * Swish
  * Tanh
  * Sigmoid
To understand and follow the thesis of the textbooks, I will first compare MAE and R² for each activation function. In the second step, I will compare: error distribution, loss progression, prediction vs reality, and residual plots.

# Results
After running my Python-script, I've got the following results. They are ordered by ascending R²-values.

| Activiation-function | MAE   | R²-Score |
|----------------------|-------|----------|
| ELU                  | 0,259 | 0,849    |
| SeLU                 | 0,270 | 0,837    |
| Swish                | 0,284 | 0,815    |
| Tanh                 | 0,262 | 0,815    |
| ReLU                 | 0,304 | 0,802    |
| Leaky_ReLU           | 0,320 | 0,793    |
| GELU                 | 0,332 | 0,767    |
| Sigmoid              | 0,482 | 0,419    |

Theory says: MAE is more robust to outliers and gives a directly interpretable average deviation. R² is useful for communicating explained variance, but it can be misleading if looked at on its own: a high R² doesn’t automatically mean good predictions on new data, and R² can even be negative if a model performs worse than just predicting the mean.
Just by viewing MAE and R²-score for each activation-function, it can be postulated: for the chosen architecture and training configuration, ELU achieved the highest R² score. Sigmoid performed significantly worse than all other activation funtions. Interstingly, GELU did not outperform ReLU in this experiment, despite its popularity in modern deep learning architectures.
The results suggest that activation-function performance is highly dependent on the dataset and network architecture.

For a more detail analysis, I've plotted loss-progession, prediction vs. reality, residual-plots and error-histogram for each evaluated activation-function:

## loss-progression
<img width="800" height="500" alt="loss_tanh" src="https://github.com/user-attachments/assets/78f74647-f035-401d-a924-33cd0e823084" />
<img width="800" height="500" alt="loss_swish" src="https://github.com/user-attachments/assets/41cd16cd-5e5e-4b6c-997a-f5ba973edebe" />
<img width="800" height="500" alt="loss_sigmoid" src="https://github.com/user-attachments/assets/25b1b39f-c5d8-43ee-9c4a-271ca2929050" />
<img width="800" height="500" alt="loss_selu" src="https://github.com/user-attachments/assets/95d58f69-e1ed-45cd-bb6d-ba7fba4333b7" />
<img width="800" height="500" alt="loss_relu" src="https://github.com/user-attachments/assets/615555c8-1b3f-42fa-ac7f-ae2eb2d9ad43" />
<img width="800" height="500" alt="loss_leaky_relu" src="https://github.com/user-attachments/assets/3533bc3e-4e1d-445e-bdec-904ce97243e9" />
<img width="800" height="500" alt="loss_gelu" src="https://github.com/user-attachments/assets/33c2db54-1376-4e45-9bef-31d8ae101443" />
<img width="800" height="500" alt="loss_elu" src="https://github.com/user-attachments/assets/e5d02bbb-aa84-46e8-bf37-85106da57988" />

## prediction vs. reality

<img width="600" height="600" alt="pred_tanh" src="https://github.com/user-attachments/assets/f1795b7a-f88f-41a3-9011-f8b4ee2d65a5" />
<img width="600" height="600" alt="pred_swish" src="https://github.com/user-attachments/assets/49236655-710c-444b-9a8d-fe9a6f89989a" />
<img width="600" height="600" alt="pred_sigmoid" src="https://github.com/user-attachments/assets/2f1b9429-fca3-4c31-8d91-b45fb464bed9" />
<img width="600" height="600" alt="pred_selu" src="https://github.com/user-attachments/assets/6840749c-b60a-4bf3-9a94-75dac6dec8c2" />
<img width="600" height="600" alt="pred_relu" src="https://github.com/user-attachments/assets/c4249777-a321-4287-902d-1eb5c900fc73" />
<img width="600" height="600" alt="pred_leaky_relu" src="https://github.com/user-attachments/assets/d9b022c1-9a18-41a1-bd56-85bfa903e519" />
<img width="600" height="600" alt="pred_gelu" src="https://github.com/user-attachments/assets/b2893f07-4e3e-461f-9674-3a9655450180" />
<img width="600" height="600" alt="pred_elu" src="https://github.com/user-attachments/assets/ce757244-e052-4533-a923-dca6d67d7afd" />

## error history

<img width="800" height="400" alt="error_hist_tanh" src="https://github.com/user-attachments/assets/96e64fbb-4726-45ad-8abb-965dad9c7e86" />
<img width="800" height="400" alt="error_hist_swish" src="https://github.com/user-attachments/assets/884f35df-d169-46b6-92e8-5056112aa127" />
<img width="800" height="400" alt="error_hist_sigmoid" src="https://github.com/user-attachments/assets/478977fc-17c9-4ceb-bac8-d3af05125b2e" />
<img width="800" height="400" alt="error_hist_selu" src="https://github.com/user-attachments/assets/304d288e-a72f-4ce1-ab8a-90db661c9a1f" />
<img width="800" height="400" alt="error_hist_relu" src="https://github.com/user-attachments/assets/9ca0c224-f15c-47f0-9360-d41b2de51aa8" />
<img width="800" height="400" alt="error_hist_leaky_relu" src="https://github.com/user-attachments/assets/c2d9c67c-8a39-42ea-927e-9aa54ca84b8f" />
<img width="800" height="400" alt="error_hist_gelu" src="https://github.com/user-attachments/assets/d4bdcee7-be91-4c80-9f9a-edd264498e00" />
<img width="800" height="400" alt="error_hist_elu" src="https://github.com/user-attachments/assets/5111f210-20c6-40fd-a84f-5ba12c7545cb" />

## residual-plot

<img width="800" height="400" alt="residual_tanh" src="https://github.com/user-attachments/assets/2a4d5f2c-e03a-4f0a-b32f-a13a9e350a4a" />
<img width="800" height="400" alt="residual_swish" src="https://github.com/user-attachments/assets/1760e943-a352-4c3e-8f2e-33151ba4559e" />
<img width="800" height="400" alt="residual_sigmoid" src="https://github.com/user-attachments/assets/12de109b-efa2-47af-b2d0-befa5003ca16" />
<img width="800" height="400" alt="residual_selu" src="https://github.com/user-attachments/assets/cc230429-ed63-41c5-b296-2ad0b4d20964" />
<img width="800" height="400" alt="residual_relu" src="https://github.com/user-attachments/assets/fb92db62-2621-418f-8b48-363832233bb9" />
<img width="800" height="400" alt="residual_leaky_relu" src="https://github.com/user-attachments/assets/000fda07-fe01-4543-a51c-002158bacf48" />
<img width="800" height="400" alt="residual_gelu" src="https://github.com/user-attachments/assets/418eb1c7-7635-4fd6-aacf-4cdb652b11b9" />
<img width="800" height="400" alt="residual_elu" src="https://github.com/user-attachments/assets/b1ef1a58-9f8f-4363-9fde-4359429febd2" />




# Next steps / experiments
  * Change of layers of the model
  * change of random_state --> 10 runs -->  average --> standard deviation
