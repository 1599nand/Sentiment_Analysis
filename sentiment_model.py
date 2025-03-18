import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary resources

# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters (special characters)
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = word_tokenize(text)  # Tokenization: Splits sentence into words
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization (corrected typo)
    return ' '.join(words)  # Join words back into a cleaned sentence

# Read dataset
data = pd.read_csv('amazon_review.csv')

# Select relevant columns
useful_data = data[['reviews.rating', 'reviews.text']].copy()

# Apply text preprocessing
useful_data['cleaned_text'] = useful_data['reviews.text'].astype(str).apply(preprocess_text)
useful_data['label']=useful_data['reviews.rating'].apply(lambda x:1 if x>3 else 0)

# Print first 5 rows of pr
print(useful_data.head())


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(useful_data['cleaned_text'])
y=useful_data['reviews.rating'].apply(lambda x:1 if x>3 else 0)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=MultinomialNB()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f'Model Accuracy:{accuracy:2f}')

# test_reviews = ["Worst product ever!", "I really like this.", "Not bad, but could be better."]

# # Convert list to 2D array
# test_reviews_cleaned= [preprocess_text(review) for review in test_reviews]
# test_review_tfidf= vectorizer.transform(test_reviews_cleaned)
# predictions = model.predict(test_review_tfidf)

# for i in predictions:
#     if i>0:
#         print('positive')
#     else:
#         print('negative')


from sklearn.linear_model import LogisticRegression

# Initialize and train Logistic Regression model
model = LogisticRegression(max_iter=1000,solver='liblinear',C=1)  # Increase iterations for better convergence
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')




#LSTM MODEL

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,LSTM,Dropout
from sklearn.model_selection import train_test_split


X=useful_data['cleaned_text'].values
y=useful_data['label'].values

print("Label distribution:\n", useful_data['reviews.rating'].value_counts())

from sklearn.utils import resample

# Separate majority and minority classes
majority = useful_data[useful_data['label'] == 1]
minority = useful_data[useful_data['label'] == 0]

# Resample to balance classes
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

# Combine back
balanced_data = pd.concat([majority, minority_upsampled])

# Check new label distribution
print("Balanced label distribution:\n", balanced_data['label'].value_counts())



vocab_size=5000
max_length=200
#Tokenization
tokenizer=Tokenizer(num_words=vocab_size,oov_token='<OOV>')
tokenizer.fit_on_texts(X)
X_sequences=tokenizer.texts_to_sequences(X)


X_padded=pad_sequences(X_sequences,maxlen=max_length,padding='post',truncating='post')
X_train,X_test,y_train,y_test=train_test_split(X_padded,y,test_size=0.2,random_state=42)

model=Sequential([
    Embedding(input_dim=vocab_size,output_dim=128,input_length=max_length),
    LSTM(64,return_sequences=True),
    Dropout(0.3),
    LSTM(32,return_sequences=True),
    LSTM(16),
    Dense(16,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

history=model.fit(X_train,y_train,epochs=5,batch_size=32,validation_data=(X_test,y_test))
print(history.history)

y_pred=(model.predict(X_test)>0.5).astype(int)
accuracy=accuracy_score(y_test,y_pred)
print(f'Model Accuracy:{accuracy:.4f}')


test_reviews = ["Worst product ever!", "I really like this.", "Not bad, but could be better."]

# Preprocess and tokenize test reviews
test_reviews_cleaned = [preprocess_text(review) for review in test_reviews]
test_review_seq = tokenizer.texts_to_sequences(test_reviews_cleaned)
test_review_padded = pad_sequences(test_review_seq, maxlen=max_length, padding='post')

# Predict sentiment
predictions = (model.predict(test_review_padded) > 0.5).astype("int32")

# Print results
for review, sentiment in zip(test_reviews, predictions):
    print(f"Review: {review} | Sentiment: {'Positive' if sentiment[0] == 1 else 'Negative'}")
