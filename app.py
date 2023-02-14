import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import pickle

vectorizer = pickle.load(open('vec.p','rb'))

model=pickle.load(open('model.p','rb'))


st.title("Sentiment Analysis for Youtube Comments")

sentence = st.text_input("input you comment here:")
if sentence:
    result = model.predict(vectorizer.transform([sentence]))
    if result == 0:
        st.write("The comment is Real")
    else:
        st.write("The comment is Fake")

    
