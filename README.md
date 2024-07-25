# Message_Chatbot

## Purpose
This script demonstrates creating a model using a data set of question-answer pairs extracted from individual Facebook conversations in Vietnamese that are preprocessed. from Hugging Face and use it to automatically reply to messages on Facebook.

## Function
- The data set is created from personal Facebook chat logs in Vietnamese.
- Model creation is pre-processed with model data that has been pre-trained on a data set of question-answer pairs.
- Use selenium to automatically reply to conversations

## Analysis
1. **Extract data**:
 - Download personal information from Facebook account.
 - Use Python to extract and pre-process conversations from downloaded Facebook data to create a refined dataset.
- Preprocessing data

2. **Training**:
 - Script to customize the model:
 - Complete dataset over 10 epochs.

## Communicate
- **CPU device**:
 - Please note that training on CPU may result in long waiting times due to the computational intensity of model fine-tuning.

## Use
1. **Prepare data**:
 - Make sure your dataset of question-answer pairs is saved in `fb_messages.txt`.

2. **Run fine-tuning**:
 - Execute the script to start the adjustment process.
- The code includes: data creation, training, use, and automatic application with elenium

3. **Create feedback**:
 - Use the refined model to generate answers to new questions using the code provided.
