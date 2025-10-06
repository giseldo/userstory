import re
import spacy
import joblib
import os

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

if __name__ == "__main__":
    exemplo_titulo = "Implementar tela de login"
    exemplo_descricao = "Como usuário, quero acessar o sistema de forma segura para proteger meus dados."
    contexto = preprocess_text(exemplo_titulo, exemplo_descricao)

    vec = joblib.load(os.path.join('models','cross_project.vec'))
    reg = joblib.load(os.path.join('models','cross_project.model'))

    X_bow = vec.transform([contexto])
    storypoints_pred = reg.predict(X_bow)
    print(f"Story points previstos para o exemplo: {storypoints_pred[0]:.2f}") 