# NLP Project
# Team 2: Acuna, W., Gallo, C., Ostrovsky, S.
# Question Answering System using DistilBERT
# October 21th 2023

# Introduction 
This project involves building a Question Answering System using the DistilBERT model from the Hugging Face Transformers library. The system was designed to answer questions based on a given context, making it suitable for various applications such as information retrieval and chatbots. The pre-trained model was fine-tuned on the Standford Question Answering Dataset (SQuAD), which is available on Kaggle (Standford university, n.d.), and includes about 98000 question-answer pairs extracted from more than 500 Wikipidia articles. However, considering the amount of time required to train the model on the complete dataset, we adopted a smaller portion of data, with 10000 training and 1200 validation samples. Preprocessing, model training and evaluation was implemented though different functions, starting from the recommendations and procedures provided by author authors (Apostolopoulou, A., n.d.; Ding, 2023; Fine-tuning with custom datasets, n.d.; McCormick and Ryan, 2019; Tran, n.d.). The final model performance was quantified in terms of exact match accuracy and F1-score, which were found equal to 54% and 44.56%. These results demonstrate the model’s ability to provide reasonably accurate answers to a variety of questions, although the general model’s response could be further improved, for instance, by increasing the size of the training set, tuning the hyperparameters, and exploiting larger models. 

# Requirements
In order to run this code, you need Python 3.10 installed on your computer. Also, it will be necessary to load the SQuAD dataset from Kaggle (Standford university, n.d.), including both the training (train-v1.1.json) and validation (dev-v1.1.json) sets, and install the following libraries, which are used throughout the code:

[json] - https://docs.python.org/3/library/json.html
[pathlib] - https://docs.python.org/3/library/pathlib.html
[pytorch] - https://pytorch.org/get-started/locally/
[transformers] - https://pypi.org/project/transformers/
[pandas] - https://pandas.pydata.org/
[matplotlib] - https://matplotlib.org/
[numpy] - https://numpy.org/doc/stable/user/absolute_beginners.html

Once the dataset is saved in the same model directory and all the necessary libraries are installed, the model can be run without requiring additional data from the user. 

# References

Apostolopoulou, A. (n.d.). BERT-based-pretrained-model-using-SQuAD-2.0-dataset/Fine_Tuning_Bert.ipynb, Github. https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb

Ding, S. (2023, January 3). [Fine Tune] Fine Tuning BERT for Sentiment Analysis. Medium. https://medium.com/@xiaohan_63326/fine-tune-fine-tuning-bert-for-sentiment-analysis-f5002b08f10a

Fine-tuning with custom datasets. (n.d.). Hugging Face. Retrieved October 20, 2023, from https://huggingface.co/transformers/v4.3.3/custom_datasets.html?highlight=fine%20tune

McCormick, C., & Ryan, N. (2019, July 22). BERT Fine-Tuning Tutorial with PyTorch. Chris McCormick. https://mccormickml.com/2019/07/22/BERT-fine-tuning/

Standford University. (n.d.). Standford Question Answering Dataset. Kaggle. https://huggingface.co/transformers/v4.3.3/custom_datasets.html?highlight=fine%20tune

Tran, C. (n.d.). Tutorial: Fine tuning BERT for Sentiment Analysis. Skim AI. https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

