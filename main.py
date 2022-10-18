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
    "strong left": 0,
    "soft left": .25,
    "neutral": .5,
    "soft right": .75,
    "strong right": 1,
}

questions_df = pd.read_csv('data/questions.csv')

st.markdown('# Questions')
question_sliders = list()
for i, row in questions_df.iterrows():
    st.markdown(f"**{row['question']}**")
    options_text = f"{row['0']} | {row['1']}"
    question_sliders.append(st.select_slider(options_text, options=["strong left", "soft left", "neutral", "soft right", "strong right"], value="neutral"))
    '---'

html = "https://mycreativetype.com/the-creative-types/"   
if st.button('Results'):
    answer = [mapping.get(x, 0) for x in question_sliders]
    pred = neigh.predict_proba([answer])[0]
    classes = np.array([x.capitalize() for x in neigh.classes_])
    
    order = np.argsort(pred)
    pred = pred[order]
    classes = classes[order]
  
    fig, ax = plt.subplots()
    ax.barh(classes, pred*100, color="#85007e")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    for i in range(len(pred)):
        if pred[i] > 0:
            label = f"{pred[i]:.0%}"
            plt.annotate(label, xy=(pred[i]*100 + 0.2, classes[i]), ha='left', va='center')
    
    st.pyplot(fig)
    
    components.iframe(html, height=600, scrolling=True)
    
