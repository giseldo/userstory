import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
import re
import spacy
import joblib
import os

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
stopwords = nlp.Defaults.stop_words

def preprocess_text(text):
	text = re.sub(r"www\S+|http\S+|@\S+|\s+", " ", text)
	doc = nlp(text)
	tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in stopwords]
	return ' '.join(tokens)

def generate_model(project_name):
	csv_path = os.path.join('dataset', 'neo', 'csv', f"{project_name}.csv")
	df = pd.read_csv(csv_path)
	df.dropna(inplace=True)
	# df = df[df.storypoints.between(df.storypoints.quantile(.05), df.storypoints.quantile(.95))]
	df["context"] = df["title"] + ". " + df["description"]
	df["context"] = df["context"].apply(preprocess_text)
	X = df["context"]
	y = df["storypoints"]
	vec = TfidfVectorizer()
	X_bow_matrix = vec.fit_transform(X)
	reg = SVR()
	reg.fit(X_bow_matrix, y)
	os.makedirs('models', exist_ok=True)
	joblib.dump(vec, os.path.join('models', f'{project_name}.vec'))
	joblib.dump(reg, os.path.join('models', f'{project_name}.model'))

if __name__ == "__main__":
	# project_name_list = [7764]
	project_name_list = [7776928]

	#project_name_list = [10152778,10171263,10171270,10171280,10174980,12450835,12584701,12894267,1304532,14052249,14976868,15502567,1714548,19921167,2009901,21149814,23285197,250833,2670515,28419588,28644964,28847821,3828396,3836952,4456656,5261717,6206924,7071551,7128869,734943,7603319,7764,7776928]
	
	for project_name in project_name_list:
		generate_model(str(project_name))