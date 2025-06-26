The Code in this reppsitory is focused on building a binary email spam classifier using a BERT-based model. 
The goal is to identify whether a given email is spam or not spam using natural language processing and deep learning techniques. 
The code is written in Python and uses the Hugging Face Transformers library, along with scikit-learn, NLTK, and other common tools. 
The dataset used for this project is sourced from Kaggle, the "Email Spam Classification Dataset". 
The Dataset can be found at the URL https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset/data. 
The dataset is automatically downloaded using Kaggle API credentials and unzipped into a working directory.

The code begins by installing required libraries and importing all necessary Python packages.
It then loads the dataset, performs initial data checks, and displays statistics about the dataset including shape, null values, and class distribution.
Preprocessing is applied to clean the email text. This includes converting text to lowercase, removing punctuation and numeric characters, tokenizing the text, removing stopwords, and lemmatizing the remaining words.
The cleaned text is added to the dataset along with a new column for the length of each email.

Exploratory data analysis is then conducted to visualize label distribution, word count histograms, and generate WordClouds for each class.
A correlation heatmap is plotted between the email length and the label. 
Dataset is split into training and testing sets using stratified sampling to preserve label distribution.

The training and testing sets are converted into the Hugging Face `Dataset` format.
A tokenizer corresponding to the model `bert-base-uncased` is loaded and applied to the datasets.
A pre-trained BERT model for sequence classification is also loaded and configured with two output labels. 
Training parameters are defined using the `TrainingArguments` class from Hugging Face, including batch size, learning rate, weight decay, and logging steps.

The training is performed using the Hugging Face `Trainer` API.
An accuracy metric is loaded using the `evaluate` library, and a custom function is defined to compute this metric from model predictions.
After training, the model is evaluated on the test dataset. Evaluation results including accuracy are printed, and a confusion matrix is generated to visualize the prediction performance.

In addition to evaluation metrics, a few example test samples are printed along with their predicted and actual labels for reference. 
A full classification report is also generated using scikit-learn to summarize precision, recall, f1-score, and support for each class.

The entire workflow is executed in Google Colab.
The user can switch to other transformer models such as `roberta-base` by modifying the model name in the code.

License
This project is for academic and educational purposes only. Please check individual libraries for licensing.
