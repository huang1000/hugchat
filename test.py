import streamlit as st
import pandas as pd, numpy as np
import altair as alt

def style_number(v, props=''):
    try:
        return props if v<0 else None
    except:
        pass

np.random.seed(0)
df = pd.DataFrame(np.random.randn(100,4), columns=list('ABCD'))

st.header('Data EDA')
st.write('Data analysis :sunglasses:')

sidebar = st.sidebar.selectbox("choose a column", df.columns)

st.metric('Median', round(df[sidebar].median(),2), round(df[sidebar].std(),2))
st.write(df.head().style.applymap(style_number, props='color:red;'))

c = alt.Chart(df).mark_circle().encode(
     x='A', y='B', size='C', color='D', tooltip=['A', 'B', 'C', 'D'])
st.write(c)

if st.button('say hello'):
    st.write("Hello world!")
else:
    st.write("Goodbye!")
