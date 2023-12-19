# https://legibilidade.com/sobre

import streamlit as st
import textdescriptives as td
import spacy
from spacy import displacy
import numpy as np
from openai import OpenAI
import joblib
import os
from dotenv import load_dotenv
#import spacy_streamlit

load_dotenv()

nlp =spacy.load("en_core_web_sm")

st.title("NEO USER STORY TUTOR")
#openai_key = st.sidebar.text_input("OPENAI KEY")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
st.write("A tool to help teams that use agile practices to building better User Stories")

with st.form(key="frm_principal"):
    txtuser = st.text_area(label="User Story Description", value="As a UI designer, I want to redesign the Resources page, so that it matches the new Broker design styles.")
    btn_submit = st.form_submit_button(label="Analyze")
    
if btn_submit:
    
    # models = ["en_core_web_sm"]
    # spacy_streamlit.visualize(models, txtuser)
    
    stLeg, stR, stEE, stNER, stD = st.tabs(["Readability", "Recommendation", "Estimate", "Named Entity", "Data" ])
    df = td.extract_metrics(text=txtuser, spacy_model="en_core_web_sm", metrics=None)
    
    with stNER:
        doc = nlp(txtuser) 
        dep_svg = displacy.render (doc, style="dep", jupyter=False)
        #st.header("Dependency visualizer")
        #st.image(dep_svg, width=50, use_column_width="never")
        
        st.subheader("Entity visualizer")
        ent_html = displacy.render(doc, style="ent", jupyter=False)
        st.markdown(ent_html, unsafe_allow_html=True)
        st.warning("Este módulo marca substantivos próprios da User Story informada acima. No fufuro pretende-se anotar palavras sensiveis ao contexto relacionadas a user story tais como. As a <role> I can <capability>, so that <receive benefit>.")
        
    with stLeg:
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
        st.warning("Este módulo apresenta os indíces de legibilidade mais comuns extraídos do texto da sua User Story.")
        st.warning("Readability indices need to be interpreted with caution, as their formulas use only 2 (two) variables: complex words and long sentences. Therefore, they are not able to measure the cohesion and coherence of a business User Story, which covers semantic, syntactic and pragmatic factors.")
         
    with stEE:
        st.subheader("Estimated Story Points" )
        st.selectbox("Machine Learning Predictor Model Used",("7764",), help="Utilize o modelo 7764. A escolha automática do melhor modelo a ser utilizado (ou seja qual o melhor modelo que é mais semelhante a sua User Story) ainda será implementado!")
        model = joblib.load("models/7764.model")
        vec = joblib.load("models/7764.vec")
        X_bow_matrix = vec.transform([txtuser])
        sp = model.predict(X_bow_matrix)        
        st.metric(label="Estimated Story Points", value=round(sp[0],2))
        
        st.warning("Este módulo estima automáticante o esforço em Story Poits da User Story informada acima. Esse modelo é treinado com os dados históricos de outros projetos (ex: 7765).")
        
    with stR:
        st.subheader("Recommendation" )
       
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a scrum master, skilled in create better user story for agile software projects."},
            {"role": "user", "content":"How can i improve this this user story : {}".format(txtuser)}]
        )
        st.success(completion.choices[0].message.content)
        st.warning("Este módulo traz recomendações para melhorar a escrita da sua User Story.")
            
    with stD:
        st.subheader("Basic Text Features Extracted")
        st.warning("*Not to be used")
        st.dataframe(df.T)
        