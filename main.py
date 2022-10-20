import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.neighbors import KNeighborsClassifier

done = False

# reading data
header = [f'Q{n}' for n in range(15)] + ['type']
results_df = pd.read_csv('data/results.csv', sep=";", names=header)
results_df.head()

# modeling
X = results_df.drop('type', axis=1).values
y = results_df.type.values

neigh = KNeighborsClassifier(n_neighbors=200, algorithm='brute')
neigh.fit(X, y)

# building questions
st.markdown('# Questions')
questions_df = pd.read_csv('data/questions.csv')
options = ["<<", " ", "  ",  "   ", ">>"]
mapping = {k:v for k, v in zip(options, np.linspace(0, 1, len(options)))}
question_sliders = list()
for i, row in questions_df.iterrows():
    '---'
    mapping[row["0"]] = 0
    mapping[row["1"]] = 1
    
    options[0] = row["0"]
    options[-1] = row["1"]
    
    question_sliders.append(st.select_slider(f"{row['question']}", options=options, value=options[2]))
'---'

# showing results
if st.button('Results'):
    answer = [mapping.get(x, None) for x in question_sliders]
    probabilities = neigh.predict_proba([answer])[0]
    types = np.array([x.capitalize() for x in neigh.classes_])
    
    # sorting from highest to lowest type probability 
    order = np.argsort(probabilities)
    probabilities = probabilities[order]
    types = types[order]
  
    # plotting
    fig, ax = plt.subplots()
    ax.set_title("Your creative type", loc='left')
    ax.barh(types, probabilities*100, color="#85007e")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ## annotations
    for i in range(len(probabilities)):
        if probabilities[i] > 0:
            label = f"{probabilities[i]:.0%}"
            plt.annotate(label, xy=(probabilities[i]*100 + 0.2, types[i]), ha='left', va='center')
    
    st.pyplot(fig)
    done = True
    # info link
    "https://mycreativetype.com/the-creative-types/"  
    # components.iframe("https://mycreativetype.com/the-creative-types/"  , height=600, scrolling=True)
    
