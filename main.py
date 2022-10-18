import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

header = [f'Q{n}' for n in range(15)] + ['type']
results_df = pd.read_csv('data/results.csv', sep=";", names=header)
results_df.head()

X = results_df.drop('type', axis=1).values
y = results_df.type.values

neigh = KNeighborsClassifier(n_neighbors=30, algorithm='brute')
neigh.fit(X, y)

mapping = {
    -2: 0,
    -1: .25,
    -0: .5,
    1: .75,
    2: 1,
}

questions_df = pd.read_csv('data/questions.csv')

st.markdown('# Questions')
question_sliders = list()
for i, row in questions_df.iterrows():
    st.markdown(f"**{row['question']}**")
    options_text = f"{row['0']} | {row['1']}"
    question_sliders.append(st.slider(options_text, min_value=-2, max_value=2, step=1, value=0))
    '---'

html = "https://mycreativetype.com/the-creative-types/"   
components.iframe(html)
if st.button('Results'):
    answer = [mapping.get(x, 0) for x in question_sliders]
    pred = neigh.predict_proba([answer])[0]
    classes = np.array([x.capitalize() for x in neigh.classes_])
    order = np.argsort(pred)
  
    fig, ax = plt.subplots()
    ax.barh(classes, pred*100)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    st.pyplot(fig)
    
