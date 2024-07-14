import re
import string
import numpy as np
import pandas as pd

from typing import List

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# before running the program, ensure you have downloaded NLTK resources below
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


# preprocessing functions
def lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_non_textual(text: str) -> str:
    return re.sub(r'[^a-zA-Z\s]', '', text)


def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return ' '.join([word for word in words if word.lower() not in stop_words])


def stem_text(text: str) -> str:
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in words])


def lemmatize_text(text: str) -> str:
    def get_wordnet_pos(word):
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word)) for word in words])


# apply preprocessing steps
def preprocess_data(data: pd.DataFrame, steps: List[callable]) -> pd.DataFrame:
    processed_data = data.copy()
    for step in steps:
        processed_data['body'] = processed_data['body'].apply(step)
    return processed_data


# load the GloVe embeddings
def load_glove_embeddings(file_path: str) -> dict:
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


# create feature vectors using GloVe embeddings
def create_glove_features(documents: pd.Series, embeddings_index: dict) -> np.ndarray:
    def document_vector(doc):
        words = [word for word in word_tokenize(doc) if word in embeddings_index]
        return np.mean([embeddings_index[word] for word in words], axis=0) if words else np.zeros(100)
    return np.array([document_vector(doc) for doc in documents])


# evaluate the models
def evaluate_model(X: np.ndarray, y: np.ndarray, model) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }


# load the dataset
data = pd.read_csv('CEAS_08.csv')

# load GloVe embeddings
glove_file_path = 'glove.6B.100d.txt'
embeddings_index = load_glove_embeddings(glove_file_path)

# define the classifiers
classifiers = {
    'SVM': SVC(kernel='linear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

# preprocessing combinations to test
preprocessing_combinations = [
    [lowercase],
    [stem_text],
    [lemmatize_text],
    [lowercase, stem_text],
    [lowercase, lemmatize_text],
    [lowercase, remove_punctuation],
    [lowercase, remove_punctuation, stem_text],
    [lowercase, remove_punctuation, lemmatize_text],
    [lowercase, remove_stopwords],
    [lowercase, remove_stopwords, stem_text],
    [lowercase, remove_stopwords, lemmatize_text],
    [lowercase, remove_non_textual],
    [lowercase, remove_non_textual, stem_text],
    [lowercase, remove_non_textual, lemmatize_text],
    [lowercase, remove_punctuation, remove_stopwords],
    [lowercase, remove_punctuation, remove_stopwords, stem_text],
    [lowercase, remove_punctuation, remove_stopwords, lemmatize_text]
]

# initialize vectorizers for BoW and TF-IDF
vectorizers = {
    'BoW': CountVectorizer(max_features=5000),
    'TF-IDF': TfidfVectorizer(max_features=5000)
}

# evaluate results
results = []
for steps in preprocessing_combinations:
    preprocessed_data = preprocess_data(data, steps)

    # GloVe features
    glove_features = create_glove_features(preprocessed_data['body'], embeddings_index)

    # vectorizer features: BoW and TF-IDF
    vectorizer_features = {}
    for vec_name, vectorizer in vectorizers.items():
        vectorizer_features[vec_name] = vectorizer.fit_transform(preprocessed_data['body']).toarray()

    # evaluate classifiers
    for model_name, model in classifiers.items():
        glove_results = evaluate_model(glove_features, preprocessed_data['label'], model)

        bow_results = evaluate_model(vectorizer_features['BoW'], preprocessed_data['label'], model)
        tfidf_results = evaluate_model(vectorizer_features['TF-IDF'], preprocessed_data['label'], model)

        results.append({
            'steps': [step.__name__ for step in steps],
            'model': model_name,
            'glove': glove_results,
            'bow': bow_results,
            'tfidf': tfidf_results
        })

# print results
for result in results:
    print(f"Preprocessing steps: {result['steps']} - Model: {result['model']}")
    print(f"GloVe - Accuracy: {result['glove']['accuracy']:.4f}, Precision: {result['glove']['precision']:.4f}, Recall: {result['glove']['recall']:.4f}, F1 Score: {result['glove']['f1_score']:.4f}")
    print(f"BoW - Accuracy: {result['bow']['accuracy']:.4f}, Precision: {result['bow']['precision']:.4f}, Recall: {result['bow']['recall']:.4f}, F1 Score: {result['bow']['f1_score']:.4f}")
    print(f"TF-IDF - Accuracy: {result['tfidf']['accuracy']:.4f}, Precision: {result['tfidf']['precision']:.4f}, Recall: {result['tfidf']['recall']:.4f}, F1 Score: {result['tfidf']['f1_score']:.4f}")
    print()
