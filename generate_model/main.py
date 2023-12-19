import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import re
import spacy
import joblib
import os

def generate_model(project_name): 
	nlp = spacy.load("en_core_web_sm")
	nlp.select_pipes(disable=["ner", "parser"])
	stopwords = nlp.Defaults.stop_words
	df = pd.read_csv(os.path.join('dataset', os.path.join('neo',os.path.join('csv',"{}.csv".format(project_name)))))
	df.dropna(inplace=True)
	df = df[df.storypoints.between(df.storypoints.quantile(.05), df.storypoints.quantile(.95))]

	df["context"] = df["title"] + ". " + df["description"]
 
	df["context"] = df["context"].apply(lambda x: re.sub("www\S+|http\S+|@\S+|\s+", " ", x)) # Re
    
	def tokens(text):
		doc = nlp(text)
		tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords]
		return ' '.join(tokens)
	df["context"] = df["context"].apply(tokens)

	def to_lower(text):
		return text.lower()
	df.context = df.context.apply(to_lower)

	X = df["context"]
	y = df["storypoints"]
	vec = TfidfVectorizer()
	X_bow_matrix = vec.fit_transform(X)

	reg = Ridge()
    
	reg.fit(X_bow_matrix, y)
 
	joblib.dump(vec, os.path.join('models','{}.vec'.format(project_name)))
    
	joblib.dump(reg, os.path.join('models','{}.model'.format(project_name)))
    
if __name__=="__main__":
	
 	#project_name_list = [10152778,10171263,10171270,10171280,10174980,12450835,12584701,12894267,1304532,14052249,14976868,15502567,1714548,19921167,2009901,21149814,23285197,250833,2670515,28419588,28644964,28847821,3828396,3836952,4456656,5261717,6206924,7071551,7128869,734943,7603319,7764,7776928]
	project_name_list = [7764]
		
	for project_name in project_name_list:
		generate_model(str(project_name))	