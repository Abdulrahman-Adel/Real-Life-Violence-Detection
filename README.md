
# Real Life Violence Detection

## Overview

This repository contains the implementation of a deep learning model for real-life violence detection using the Vision Transformer (ViT) architecture. The model is trained on the Real Life Violence Situations Dataset, hosted on Kaggle.

<<<<<<< HEAD
## Project Structure
=======
```
├──  base
│   └── base_trainer.py   - this file contains the abstract class of the trainer.
│
│
├── models              - this folder contains the models for the project.
│   └── model_01.py
│
│
├── trainers             - this folder contains trainers of your project.
│   └── trainer.py
│   
│  
├──  data _loader  
│    └── data_loader_01.py  - data loader responsible for handling data generation and preprocessing
│
│
├── train.py --  main used to run the training across different config files and models
│
├── evaluate.py --  files responsible for the evaluation of different models. Loading and selecting the best model
│ 
└── train_bash.sh --  example bash script to run the training with different arguments
>>>>>>> 467ac69c89c87ee8347844fc4ac97639bf267059

- **`notebooks/`**: Jupyter notebooks for data exploration, model training, and evaluation.
- **`src/`**: Source code for the project.
  - **`base/`**: this folder contains the abstract class of the model and trainer.
  - **`data_loader/`**: Data preprocessing and loading scripts.
  - **`models/`**: Implementation of the Vision Transformer model.
  - **`trainers/`**: trainer class for a custom training loop.
- **`datasets/`**: Placeholder for the Real Life Violence Situations Dataset (not included in this repository).

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.13+
- Other dependencies specified in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/real-life-violence-detection.git
   cd real-life-violence-detection
   ```
2. Install dependencies:
     
     ```bash
     pip install -r requirements.txt
     ```
### Usage

1. Download the Real Life Violence Situations Dataset from Kaggle and place it in the datasets/ directory.

2. Run the Jupyter notebooks in the notebooks/ directory for data exploration, model training, and evaluation.

3. To train the model, execute:
     ```
     python src/train.py
     ```
4. Evaluate the trained model:
     ```
     python src/evaluate.py
     ```
     
## Results

Include key results, metrics, and visualizations here. Highlight the model's performance on various evaluation metrics.

## Model Architecture

Describe the Vision Transformer architecture used, along with any modifications or adaptations made for this project.

## Acknowledgments

Mention any external libraries, codebases, or research papers you referred to.

## License

This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT).

## Contact
Abdulrahman Adel Ibrahim<br>
Email: abdulrahman.adel098@gmail.com<br>
<br>
Feel free to reach out with any questions or suggestions!


 
