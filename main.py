# https://legibilidade.com/sobre

import streamlit as st
import textdescriptives as td
import spacy
from spacy import displacy
import numpy as np
#import spacy_streamlit

nlp =spacy.load("en_core_web_sm")

st.title("NEO USER STORY HELPER")

st.write("Uma ferramenta para auxiliar na construção de user story")

with st.form(key="frm_principal"):
    txtuser = st.text_area(label="Descrição da User Story", value="As a UI designer, I want to redesign the Resources page, so that it matches the new Broker design styles.")
    btn_submit = st.form_submit_button(label="Verificar Qualidade")
    
if btn_submit:
    
    # models = ["en_core_web_sm"]
    # spacy_streamlit.visualize(models, txtuser)
    
    stNER, stLeg, stEE, stR, stD = st.tabs(["NER", "Legibilidade", "Estimativa de Esforço", "Recomendação", "Dados" ])
    df = td.extract_metrics(text=txtuser, spacy_model="en_core_web_sm", metrics=None)
    
    with stNER:
        doc = nlp(txtuser) 
        dep_svg = displacy.render (doc, style="dep", jupyter=False)
        #st.header("Dependency visualizer")
        #st.image(dep_svg, width=50, use_column_width="never")
        
        st.header("Entity visualizer")
        ent_html = displacy.render(doc, style="ent", jupyter=False)
        st.markdown(ent_html, unsafe_allow_html=True)
        
    with stLeg:
        sst1, sst2, sst3, sst4 = st.columns(4)
            
        FK = value=df["flesch_kincaid_grade"]
        GF = value=df["gunning_fog"]
        ARI = df["automated_readability_index"]
        LC = value=df["coleman_liau_index"]
        RF = round(np.mean([FK, GF, ARI, LC]), 2)
        
        sst1.metric(label="FK", value=round(FK,2))
        sst2.metric(label="GF", value=round(GF,2))
        sst3.metric(label="ARI", value=round(ARI,2))
        sst4.metric(label="LC", value=round(LC,2))
        ssst1, ssst2, ssst3, ssst4 = st.columns(4)
        ssst1.metric(label="RF", value= RF )
         
    with stEE:
        FK = value=df["flesch_kincaid_grade"]
        GF = value=df["gunning_fog"]
        ARI = df["automated_readability_index"]
        LC = value=df["coleman_liau_index"]
        RF = round(np.mean([FK, GF, ARI, LC]), 2)
        #st.write("Simples ou Complexo")
        st.write("Opções possíveis: Alto, Médio ou Baixo")
        
        if RF > 12.24:
            st.warning("""Estimativa de esforço: Alto""")
        else:
            st.warning("""Estimativa de esforço: Baixo""")
    
    with stR:
        st.warning("Recomendamos aumentar a legibilidade do texto")
            
    with stD:
        st.dataframe(df.T)
        