import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
import re
import spacy
import joblib
import os
import pandas as pd

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
stopwords = nlp.Defaults.stop_words

def preprocess_text(title, description):
    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(disable=["ner", "parser"])
    stopwords = nlp.Defaults.stop_words
    contexto = title + ". " + description
    contexto = re.sub("www\S+|http\S+|@\S+|\s+", " ", contexto)
    def tokens(text):
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords]
        return ' '.join(tokens)
    contexto = tokens(contexto)
    contexto = contexto.lower()
    return contexto

def generate_model():
	df = pd.read_csv("hf://datasets/giseldo/neodataset/issues.csv")
	df.dropna(inplace=True)
	df = df[df.storypoints.between(df.storypoints.quantile(.05), df.storypoints.quantile(.95))] # remove outliers based on the 5th and 95th percentiles	
	df["context"] = df["title"] + ". " + df["description"]
	df["context"] = df["context"].apply(preprocess_text)
	X = df["context"]
	y = df["storypoints"]
	vec = TfidfVectorizer()
	X_bow_matrix = vec.fit_transform(X)
	reg = SVR()
	reg.fit(X_bow_matrix, y)
	os.makedirs('models', exist_ok=True)
	joblib.dump(vec, os.path.join('models', f'cross_project.vec'))
	joblib.dump(reg, os.path.join('models', f'cross_project.model'))

if __name__ == "__main__":
	generate_model()