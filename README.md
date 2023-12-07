
# Real Life Violence Detection

## Overview

This repository contains the implementation of a deep learning model for real-life violence detection using the Vision Transformer for video classification (ViViT) architecture. The model is trained on the <a href="https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset/">Real Life Violence Situations Dataset</a>, hosted on Kaggle.

## Project Structure

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

2. Run the Jupyter notebooks in the notebooks/ directory for data exploration.

3. To train the model, execute:

     ```
     python src/train.py
     ```

4. Evaluate the trained model:

     ```
     python src/evaluate.py
     ```

## Results

the performance of a Vision Transformer-based model for real-life violence detection, trained using Kaggle's P100 GPU gave promising descriptive statistics for the model's performance across multiple metrics. The mean accuracy across 30 epochs reached 85%, with a standard deviation of 2%. The precision and recall scores for violence detection were consistent, averaging 0.88 and 0.86, respectively.

Accuracy Curve             |      Loss Curve
:-------------------------:|:-------------------------:
<img src="assets/model%20accuracy.png"/> | <img src="assets/model%20loss%20evolotion.png"/>

## Model Architecture

The Vision Transformer (ViT) architecture, introduced by Alexey Dosovitskiy and his colleagues at Google Research, is a novel approach to computer vision tasks, particularly image classification. Unlike traditional Convolutional Neural Networks (CNNs), which have been dominant in image processing tasks, ViT uses a transformer architecture, originally designed for natural language processing tasks. Below is a detailed explanation of the Vision Transformer architecture:

<center><img src="assets\vivit arch.png" alt="ViVit arch"></center>

### 1. Video Frame Input

- Instead of processing individual images, the ViT for videos would take sequences of video frames as input.
- Video frames are divided into fixed-size non-overlapping patches similar to the original ViT for images.

### 2. Temporal Sequence

- Each patch in the sequence represents a frame in the video, and the entire sequence forms a temporal representation.
- Tokens are created for each patch, and the sequence of these tokens represents the temporal evolution of the video.

### 3. 3D Token Embedding

- To capture both spatial and temporal features, each patch is linearly embedded into a high-dimensional vector using a 3D linear projection.
- The 3D token embedding (tubelet embedding) includes spatial information within each frame and temporal information across frames.

<center><img src="assets\tubelet embedding.png" alt="tubelet"></center>

### 4. Positional Embeddings Across Frames

- Positional embeddings are added to the 3D token embeddings to encode spatial and temporal information.
- These embeddings convey both the spatial location within a frame and the temporal order across frames.
  
### 5. Transformer Encoder Blocks

<img align="right" src="assets\encoder block.png" alt="encoder block">

- The core of the ViT architecture consists of multiple layers of transformer encoder blocks.
- Each encoder block typically includes:
  - Multi-Head Self-Attention Mechanism (MSA):
    - Enables tokens to attend to different parts of the input sequence, capturing global and local dependencies.
  - Feedforward Neural Network (FFN):
    - Applies a non-linear transformation to the attended features.
  - Layer Normalization and Residual Connections:
    - Enhances the stability and training of the model.

### 6. Classification Head

- After passing through the transformer encoder blocks, the output token embeddings are used for the final classification.
- A special token (CLS token) is added at the beginning of the sequence, and its final embedding is used as a summary representation for the entire input video.
- The CLS token's embedding is then fed into a classification head, considering both spatial and temporal features.

## Next Steps

- [ ] Data Augmentation
- [ ] Hyperparamter Tuning
- [x] Learning rate Scheduler

## Acknowledgments

- <a href="https://arxiv.org/abs/2103.15691">ViViT: A Video Vision Transformer</a>
- <a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a>
- <a href="https://arxiv.org/abs/2210.07240">How to Train Vision Transformer on Small-scale Datasets?</a>
- <a href="https://arxiv.org/abs/2211.05187">Training a Vision Transformer from scratch in less than 24 hours with 1 GPU</a>

## License

This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT).

## Contact

Abdulrahman Adel Ibrahim<br>
Email: <abdulrahman.adel098@gmail.com><br>
<br>
Feel free to reach out with any questions or suggestions!
