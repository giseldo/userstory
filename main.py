# https://legibilidade.com/sobre

import streamlit as st
import textdescriptives as td
import spacy
from spacy import displacy
import numpy as np
from openai import OpenAI, AuthenticationError
import joblib
from dotenv import load_dotenv
import re

load_dotenv()

API_KEY = st.sidebar.text_input("OPENAI KEY", type="password")
if not API_KEY:
    st.sidebar.error("Please enter a valid API key to continue.")

nlp =spacy.load("en_core_web_sm")

st.title("USER STORY TUTOR")
st.write("A tool to help teams that use agile practices to estimate and build better User Stories")

# Organizando os checkboxes em colunas lado a lado
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    usellm = st.checkbox("Use LLM recommendation", False)
with col2:
    useNER = st.checkbox("Use NER", False)
with col3:
    usereadability = st.checkbox("Use Readability Indexes", False)
with col4:
    useEstimator = st.checkbox("Use Estimator Cross-project", False)
with col5:
    useBasicTextFeatures = st.checkbox("Use Basic Text Features", False)

with st.form(key="frm_principal"):
    txttitle = st.text_input(label="User Story Title", value="resource page")
    txtuser = st.text_area(label="User Story Description", value="As a UI designer, I want to redesign the Resources page, so that it matches the new Broker design styles.")
    btn_submit = st.form_submit_button(label="Analyze")

def get_nearest_fibonacci(value):
    """Retorna o número de Fibonacci mais próximo do valor fornecido"""
    fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    
    if value <= 1:
        return 1
    
    closest = fibonacci_sequence[0]
    min_diff = abs(value - closest)
    
    for fib in fibonacci_sequence:
        diff = abs(value - fib)
        if diff < min_diff:
            min_diff = diff
            closest = fib
    
    return closest

def preprocess_text(title, description):
    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(disable=["ner", "parser"])
    stopwords = nlp.Defaults.stop_words
    text = title + ". " + description
    text = re.sub(r"www\S+|http\S+|@\S+|\s+", " ", text)
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in stopwords]
    return ' '.join(tokens)
    
if btn_submit:

    stLeg, stR, stEE, stNER, stD = st.tabs(["Readability Indexes", "LLM recommendation", "Estimator Cross-project", "Named Entity", "Basic Text Features" ])
    df = td.extract_metrics(text=txtuser, spacy_model="en_core_web_sm", metrics=None)
    
    with stNER:
        if useNER:
            st.header("Named Entity Recognition - NER" )
            doc = nlp(txtuser) 
            dep_svg = displacy.render (doc, style="dep", jupyter=False)
            st.subheader("Entity visualizer")
            ent_html = displacy.render(doc, style="ent", jupyter=False)
            st.markdown(ent_html, unsafe_allow_html=True)
            st.warning("This module marks proper nouns from the User Story reported above. In the future, the aim is to annotate context-sensitive words related to the user story, such as: As a <role> I can <capability>, so that <receive benefit>.")
            
    with stLeg:
        if usereadability:
            st.header("Readability" )
            sst1, sst2, sst3, sst4 = st.columns(4)
                
            FK = value=df["flesch_kincaid_grade"]
            GF = value=df["gunning_fog"]
            ARI = df["automated_readability_index"]
            LC = value=df["coleman_liau_index"]
            RF = round(np.mean([FK, GF, ARI, LC]), 2)
            
            sst1.metric(label="FK", value=round(FK,2), help= "Teste de facilidade de leitura de Flesch")
            sst2.metric(label="GF", value=round(GF,2), help="Gunning fog index")
            sst3.metric(label="ARI", value=round(ARI,2), help="Automated Readability Index - ARI")
            sst4.metric(label="LC", value=round(LC,2), help="Coleman-Liau index")
            ssst1, ssst2, ssst3, ssst4 = st.columns(4)

            ssst1.metric(label="RF", value= RF, help="Resultado Final. Média aritmética entre os indicadores FK, GF, ARI e LC")
            st.warning("This module presents the most common readability indices extracted from your User Story text.")
            st.warning("Readability indices need to be interpreted with caution, as their formulas use only 2 (two) variables: complex words and long sentences. Therefore, they are not able to measure the cohesion and coherence of a business User Story, which covers semantic, syntactic and pragmatic factors.")
         
    with stEE:
        if useEstimator:
            st.header("Story Points Estimator")
        
            contexto = preprocess_text(txttitle, txtuser)
            model = joblib.load(f"models/cross_project.model")
            vec = joblib.load(f"models/cross_project.vec")
            X_bow_matrix = vec.transform([contexto])
            sp = model.predict(X_bow_matrix)        
            fibonacci_sp = get_nearest_fibonacci(sp[0])
            st.metric(label="Estimated Story Points", value=f"{fibonacci_sp} (original: {round(sp[0],2)})")
            st.warning("This module automatically estimates the effort in Story Points for the User Story provided above. The model is trained using historical data from other projects.")
        
    with stR:
        if usellm:
            st.subheader("LLM Recommendation")
            try:
                client = OpenAI(api_key=API_KEY)
                completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a scrum master, skilled in create better user story for agile software projects."},
                    {"role": "user", "content":"How can i improve this this user story : {}".format(txtuser)}]
                )
                st.success(completion.choices[0].message.content)
            except AuthenticationError as e:
                st.error("Authentication error: Invalid or missing API key. Please check your OpenAI key.")
            except Exception as e:
                st.error("Unexpected error communicating with OpenAI API.")
            
            st.warning("This module provides recommendations to improve the writing of your User Story.")
            
    with stD:
        if useBasicTextFeatures:
            st.header("Basic Text Features" )
            st.dataframe(df.T)