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
import glob

load_dotenv()

API_KEY = st.sidebar.text_input("OPENAI KEY", type="password")
if not API_KEY:
    st.sidebar.error("Por favor, informe uma chave de API válida para continuar")

nlp =spacy.load("en_core_web_sm")

st.title("NEO USER STORY TUTOR (UST)")

st.write("A tool to help teams that use agile practices to estimate and build better User Stories")

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
        #st.warning("Este módulo marca substantivos próprios da User Story informada acima. No fufuro pretende-se anotar palavras sensiveis ao contexto relacionadas a user story tais como. As a <role> I can <capability>, so that <receive benefit>.")
        
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
        st.warning("This module presents the most common readability indices extracted from your User Story text.")
        st.warning("Readability indices need to be interpreted with caution, as their formulas use only 2 (two) variables: complex words and long sentences. Therefore, they are not able to measure the cohesion and coherence of a business User Story, which covers semantic, syntactic and pragmatic factors.")
         
    with stEE:
        st.subheader("Estimated Story Points" )

        model_files = glob.glob("models/*.*")
        model_names = set()
        for file in model_files:
            base = os.path.basename(file)
            name = os.path.splitext(base)[0]
            model_names.add(name)
        model_names = sorted(list(model_names))
        
        st.selectbox("Machine Learning Predictor Model Used", model_names)
        selected_model = model_names[0] if model_names else None

        if selected_model:
            model = joblib.load(f"models/{selected_model}.model")
            vec = joblib.load(f"models/{selected_model}.vec")
            X_bow_matrix = vec.transform([txtuser])
            sp = model.predict(X_bow_matrix)        
            st.metric(label="Estimated Story Points", value=round(sp[0],2))
        
        st.warning("This module automatically estimates the effort in Story Points for the User Story provided above. The model is trained using historical data from other projects.")
        
    with stR:
        st.subheader("Recommendation")
       
        client = OpenAI(api_key=API_KEY)
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a scrum master, skilled in create better user story for agile software projects."},
            {"role": "user", "content":"How can i improve this this user story : {}".format(txtuser)}]
        )
        st.success(completion.choices[0].message.content)
        st.warning("Este módulo traz recomendações para melhorar a escrita da sua User Story.")
            
    with stD:
        st.subheader("Basic Text Features Extracted")
        st.dataframe(df.T)
        