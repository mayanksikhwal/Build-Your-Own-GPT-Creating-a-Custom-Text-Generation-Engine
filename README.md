# Build-Your-Own-GPT-Creating-a-Custom-Text-Generation-Engine

## Project Description
This project trains a compact GPT-2 style language model to generate short children’s stories using the **TinyStories** dataset. It covers data loading, tokenization, model configuration, custom training, checkpointing, and sampling from saved checkpoints. The goal is to demonstrate an end-to-end language model training process on a smaller scale, suitable for limited resources.

## Tech Stack
*   Python: The primary programming language used for model training and inference.
*   PyTorch: The deep learning framework used for building and training the model.
*   Transformers: Used for loading pre-trained models, tokenizers, and model configurations from Hugging Face.
*   Datasets: Used for loading and streaming the TinyStories dataset from Hugging Face.
*   TQDM: Used for displaying progress bars during training.
*   Matplotlib: Used for plotting training history (e.g., loss curves).
*   Google Colab: The environment where the notebook was developed and executed, utilizing GPU acceleration.

## How to Run
1.  **Open the notebook in Google Colab:** Upload or open the `tiny_llm_story_generator_training.ipynb` notebook in your Google Colab environment.
2.  **Mount Google Drive:** Run the cell to mount your Google Drive. This is necessary for saving model checkpoints and training history.
3.  **Install Libraries and Load Data:** Run the cell to install the required libraries and load the TinyStories dataset in streaming mode.
4.  **Define Dataset Class:** Run the cell containing the `TinyStoriesStreamDataset` class definition.
5.  **Load Tokenizer, DataLoader, Model, and Optimizer:** Run the cell to load the GPT-2 tokenizer, set up the DataLoader, configure and initialize the GPT-2 model (with a small configuration), move it to the appropriate device (GPU if available), and set up the optimizer.
6.  **Run Training Loop:** Execute the cell containing the training loop. This will train the model for the specified number of epochs, save checkpoints periodically, and generate sample text after each epoch.
7.  **Resume Training (Optional):** If you stopped training and want to resume from a checkpoint, run the cell designed for loading from a checkpoint and continuing training.
8.  **Generate Text from Saved Checkpoint:** Run the cell to load a saved checkpoint and generate text using the trained model.
9.  **Inference with Pretrained TinyStories Model (Optional):** Run the cell to perform inference using a larger, pre-trained TinyStories model from Hugging Face.
