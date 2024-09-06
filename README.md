# Sentiment Analysis Using BERT

## Project Overview

This project implements a Sentiment Analysis Model using the BERT (Bidirectional Encoder Representations from Transformers) model, combined with Natural Language Processing (NLP) and Machine Learning (ML) techniques to analyze the sentiment of tweets. By leveraging Python and powerful libraries such as NumPy, Pandas, and PyTorch, the model determines whether tweets have a positive or negative sentiment.

The dataset used is the Sentiment140 dataset, which contains 1.6 million tweets labeled with either positive or negative sentiment. The model processes this dataset and achieves an accuracy of 85%, though the performance could be improved if more computational resources were available to handle the full dataset effectively.

## Key Features
-Model: The model is built using BERT from the Hugging Face transformers library.
-Dataset: The model is trained on the Sentiment140 dataset, which consists of 1.6 million tweets. Each tweet is labeled as either positive (1) or negative (0).
-Training and Testing: The dataset is split into 80% for training and 20% for testing the model.
-Evaluation: The model's performance is evaluated using F1 score and accuracy per class (positive and negative).
-Challenges: Due to hardware limitations (running on a personal laptop), the full dataset of 1.6 million tweets could not be processed, potentially impacting the final test score.

### Project Structure
Sentiment140 Dataset: The dataset includes tweet text and sentiment labels. Tweets are preprocessed to remove URLs, hashtags, and mentions.
BERT Tokenization: Each tweet is tokenized using BERT's tokenizer, adding special tokens such as [CLS] and [SEP].
Training and Testing: The data is split into a training set and a testing set, and the model is trained using AdamW optimizer and a learning rate scheduler.
Output: The results, including the training loss, validation loss, and test scores, are saved to an output.txt file.
