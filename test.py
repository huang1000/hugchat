import streamlit as st
from streamlit_shap import st_shap
import shap
import pandas as pd, numpy as np
import altair as alt
import datetime, time
from sklearn.model_selection import train_test_split
import xgboost


st.set_page_config(layout='wide')

def style_number(v, props=''):
    try:
        return props if v<0 else None
    except:
        pass

@st.cache_data()
def load_data():
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(100,5), columns=list('ABCDE'))
    return df

st.title('[Streamlit learning notes](https://30days.streamlit.app/?challenge=Day30)')
df = load_data()
col1, col2 = st.columns(2)
with col1:
    st.write('Data analysis :sunglasses:')
    st.write("`st.metric('Median', value, value)`")
    st.metric('Median', round(df['A'].median(),2), round(df['A'].std(),2))
    st.write("`st.write(df)`")
    st.write(df.head().style.applymap(style_number, props='color:red;'))

with col2:
    st.write("`st.write(altair.Chart(df).mark_circle().encode(x='A',y='B',tooltip=list('ABCD')))`")
    cols = st.multiselect("Choose columns", df.columns)
    if len(cols)>1:
        st.write(alt.Chart(df[cols]).mark_circle().encode(
            x=cols[0], y=cols[1], size=cols[min(2,len(cols)-1)], color=cols[min(3,len(cols)-1)], tooltip=cols))

st.markdown("""---""")

with st.expander("st.progress(0), st.button('text'), st.line_chart(df)"):
    st.write("`st.altair_chart(altair_chart, use_container_width=False, theme='streamlit')`")
    st.write("""
        ```python
        c = alt.Chart(df).mark_circle().encode(
            x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

        st.altair_chart(c, use_container_width=True)
        ```
        """)
    my_bar = st.progress(0)
    if st.button('Start'):
        for pct_complete in range(100):
            time.sleep(0.05)
            my_bar.progress(pct_complete+1)
        st.line_chart(df.iloc[:,:2])
        st.balloons()


with st.expander("st.slider('title', 0,100, (25,75))"):
    st.write("`st.slider('title', 0.0, 100.0 (25.0,75.0))`")
    values = st.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0))
    st.write('Values:', values)

    appointment = st.slider(
        "Schedule your appointment:",
        value=(datetime.time(11, 30), datetime.time(14, 45)))
    st.write("You're scheduled for:", appointment)

    st.write("`start_time = st.slider('When do you start?', value=datetime(2023, 7, 28, 9, 30), \
        format='MM/DD/YY - hh:mm')`")
    start_time = st.slider(
        "When do you start?",
        value=datetime.datetime(2023, 7, 28, 9, 30),
        format="MM/DD/YY - hh:mm")
    st.write("Start time:", start_time)


with st.expander("select widget"):
    st.write("`option = st.selectbox('title', 'list_of_options')`")
    st.write("`options = st.multiselect('title', 'list_of_options')`")
    st.write("`st.checkbox('text')`")

    st.write("What do you want to order?")
    tea = st.checkbox("Tea")
    coffee = st.checkbox("Coffee")
    cola = st.checkbox("Cola")
    if tea:
        st.write("Great! Here's some more tea")
    if coffee: 
        st.write("Okay, here's some coffee ☕")
    if cola:
        st.write("Here you go 🥤")

st.markdown("""---""")
st.subheader("Streamlit Components")
st.write("`pip install streamlit_pandas_profiling`")
st.write("""
```python
from streamlit_pandas_profiling import st_profile_report
pr = df.profile_report()
st_profile_report(pr)
        """)
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

with st.expander("streamlit_pandas_profiling"):
    if len(cols)>0:
        pr = df[cols].profile_report()
        st_profile_report(pr)

st.markdown("""---""")
with st.expander("st.latex(r'text')"):
    st.latex(r'''
        a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
        \sum_{k=0}^{n-1} ar^k =
        a \left(\frac{1-r^{n}}{1-r}\right)
        ''')

with st.expander("Customizing the theme"):
    st.write('Contents of the `.streamlit/config.toml` file')
    st.code("""
    [theme]
    primaryColor="#F39C12"
    backgroundColor="#2E86C1"
    secondaryBackgroundColor="#AED6F1"
    textColor="#FFFFFF"
    font="monospace"
            """)


st.subheader("st.secrets['key']")
st.write("`st.secrets['my_cool_secrets']['message']`")


st.subheader("st.file_uploader('Choose a file')")
uploaded_file = st.file_uploader("By default, uploaded files are limited to 200MB. You can configure this using the server.maxUploadSize config option.")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown(""" ### DataFrame """)
    st.write(df)
    st.markdown(""" ### Description """)
    st.write(df.describe())
else:
    st.info('☝️ Upload a CSV file')

st.markdown("---")
st.subheader('st.form')

st.write("`st.form` creates a form that batches elements together with a \"Submit\" button.")

with st.form('my_form'):
    st.subheader('**Order your coffee**')
    coffee_bean_val = st.selectbox('Coffee bean', ['Arabica', 'Robusta'])
    coffee_roast_val = st.selectbox('Coffee roast', ['Light', 'Medium', 'Dark'])
    brew_val = st.selectbox('Coffee brewing method', ['Aeropress', 'Drip', 'French press', 'Moka pot', 'Siphon'])
    serving_type_val = st.selectbox('Serving format', ['Hot', 'Iced', 'Frappe'])
    milk_val = st.select_slider('Milk intensity', ['None', 'Low', 'Medium', 'High'])
    owncup_val = st.checkbox("Bring own cup")
    submitted = st.form_submit_button('Submit')

if submitted:
    st.markdown(f'''
        ☕ You have ordered:
        - Coffee bean: `{coffee_bean_val}`
        - Coffee roast: `{coffee_roast_val}`
        - Brewing: `{brew_val}`
        - Serving type: `{serving_type_val}`
        - Milk: `{milk_val}`
        - Bring own cup: `{owncup_val}`
                ''')
else:
    st.write('☝️ Place your order!')

st.markdown('---')
st.subheader('st.experimental_get_query_params')
st.markdown("""
In the above URL bar of your internet browser, append the following:
`?firstname=Jack&surname=Beanstalk`
            """)
st.code("""
        st.write(st.experimental_get_query_params())
        """)
firstname = st.experimental_get_query_params()['firstname'][0]
surname = st.experimental_get_query_params()['surname'][0]
st.write(f'Hello **{firstname} {surname}**!')

st.markdown('---')
st.subheader('st.cache_data()', 'st.cache_resource()')

st.code("""
    @st.cache_data()
    def load_data():
        df = pd.DataFrame(np.random.rand(1e6,5))
        return df
        """)

st.markdown('---')
st.subheader('st.session_state')
st.write("Streamlit reruns your script from top to bottom every time you interact with your app. Each reruns takes place in a blank slate: no variables are shared between runs.")
st.write("Session State is a way to share variables between reruns, for each user session. In addition to the ability to store and persist state, Streamlit also exposes the ability to manipulate state using Callbacks.")

def lbs_to_kg():
  st.session_state.kg = st.session_state.lbs/2.2046
def kg_to_lbs():
  st.session_state.lbs = st.session_state.kg*2.2046

st.write('Input')
col1, spacer, col2 = st.columns([2,1,2])
with col1:
  pounds = st.number_input("Pounds:", key = "lbs", on_change = lbs_to_kg)
with col2:
  kilogram = st.number_input("Kilograms:", key = "kg", on_change = kg_to_lbs)

st.write('Output')
st.write("st.session_state object:", st.session_state)

st.markdown('---')
st.subheader('How to use API by building the Bored API app 🏀')

import requests

st.sidebar.header('Input')
user_name = st.sidebar.text_input("What is your name?")
activity_type = st.sidebar.selectbox("Select an activity", ["education", "recreational", "social", "diy", "charity", "cooking", "relaxation", "music", "busywork"])
suggested_activity_url = f'http://www.boredapi.com/api/activity?type={activity_type}'
json_data = requests.get(suggested_activity_url)
suggested_activity = json_data.json()

c1, c2 = st.columns(2)
with c1:
  with st.expander('About this app'):
    st.write('Are you bored? The **Bored API app** provides suggestions on activities that you can do when you are bored. This app is powered by the Bored API.')
with c2:
  with st.expander('JSON data'):
    st.write(suggested_activity)

st.markdown('### Suggested activity')
st.info(suggested_activity['activity'])

col1, col2, col3 = st.columns(3)
with col1:
  st.metric(label='Number of Participants', value=suggested_activity['participants'], delta='')
with col2:
  st.metric(label='Type of Activity', value=suggested_activity['type'].capitalize(), delta='')
with col3:
  st.metric(label='Price', value=suggested_activity['price'], delta='')


number = st.sidebar.slider('Select a number:', 0, 10, 5)
st.sidebar.write('Selected number from slider widget is:', number)


st.markdown('---')
st.subheader('Streamlit-shap')
st.markdown('''[`streamlit-shap`](https://github.com/snehankekre/streamlit-shap) is a Streamlit component that provides a wrapper to display [SHAP](https://github.com/slundberg/shap) plots in [Streamlit](https://streamlit.io/). ''')

st.code('pip install streamlit streamlit-shap xgboost scikit-learn')

st.code( \
'''
import streamlit as st
from streamlit_shap import st_shap
import shap
from sklearn.model_selection import train_test_split
import xgboost
import numpy as np, pandas as pd

@st.cache_data
def load_data():
    return shap.datasets.adult()

@st.cache_data
def load_model(X,y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
   d_train = xgboost.DMatrix(X_train, label=y_train)
   d_test = xgboost.DMatrix(X_test, label=y_test)
   params = {
      "eta": 0.01,
      "objective": "binary:logistic",
      "subsample": 0.5,
      "base_score": np.mean(y_train),
      "eval_metric": "logloss",
      "n_jobs": -1,
   }
   model = xgboost.train(params, d_train, 10, evals=[(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
   return model

X,y = load_shap_data()
model = load_model(X,y)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)
with st.expander('Waterfall plot'):
    st_shap(shap.plots.waterfall(shap_values[0]), height=300)
with st.expander('Beeswarm plot'):
    st_shap(shap.plots.beeswarm(shap_values), height=300)
''')

@st.cache_data
def load_shap_data():
   return shap.datasets.adult()

@st.cache_data
def load_model(X,y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
   d_train = xgboost.DMatrix(X_train, label=y_train)
   d_test = xgboost.DMatrix(X_test, label=y_test)
   params = {
      "eta": 0.01,
      "objective": "binary:logistic",
      "subsample": 0.5,
      "base_score": np.mean(y_train),
      "eval_metric": "logloss",
      "n_jobs": -1,
   }
   model = xgboost.train(params, d_train, 10, evals=[(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
   return model

X,y = load_shap_data()
model = load_model(X,y)

with st.expander("About the data"):
    X_display, y_display = shap.datasets.adult(display=True)
    st.dataframe(X)
    st.dataframe(y)

st.write("`streamlit-shap` for displaying SHAP plot")
# compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
with st.expander('Waterfall plot'):
    st_shap(shap.plots.waterfall(shap_values[0]), height=300)
with st.expander('Beeswarm plot'):
    st_shap(shap.plots.beeswarm(shap_values), height=300)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
with st.expander('Force plot'):
    st.write('First data instance')
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_display.iloc[0,:]), height=200, width=1000)
    st.write('First thousand data instance')
    st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_display.iloc[:1000,:]), height=400, width=1000)

st.markdown('---')
def query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
with st.expander('How to make a zero-shot learning text classifier using HuggingFace and Streamlit'):
    st.write("We will create a zero-shot learning text classifier using [HuggingFace's API inference](https://huggingface.co/inference-api) and Distilbart!")
    st.write("You can classify keyphrases on-the-fly without pre-ML training: Create classifying labels, paste your keyphrases, and you're off!")
    st.write("The HuggingFace API inference has a generous free allowance up to 30k input characters per month")
    api_token = st.secrets['huggingface']['api_token']
    model_id = "valhalla/distilbart-mnli-12-3"
    # setup the labels
    label_widget = st_tags(label="",
                           text="Add labels - 3 max",
                           value=["Transactional", "Informational"],
                           suggestions=[
                               "Navigational", "Positive", "Negative", "Neutral",
                           ],
                           maxtags=3)
    # create the text area for keyphrases
    MAX_LINES = 5
    text = st.text_area("Enter keyphrases to classify", sample, height=200, key="2", 
                        help="At least 2 phrases for the classifier to work, one per line, " + str(MAX_LINES) + " phrases max")
    lines = text.split("\n")
    output = []
    for row in lines:
        payload = {"inputs":row, 
                   "parameters": {"candidate_labels": label_widget},
                   "options": {"wait_for_jodel": True}}
        output.append(query(payload, model_id, api_token))
    out_df = pd.DataFrame.from_dict(output)
    st.dataframe(out_df)


st.markdown('---')
def get_ytid(input_url):
  if 'youtu.be' in input_url:
    ytid = input_url.split('/')[-1]
  if 'youtube.com' in input_url:
    ytid = input_url.split('=')[-1]
  return ytid

with st.expander("Real-world problem"):
    st.write('This app retrieves the thumbnail image from a YouTube video.')
    st.write("accept a YouTube URL as input > extract video ID from the URL > retrieve the thumbnail image")

    yt_url = st.text_input('Paste YouTube URL', 'https://youtu.be/JwSS70SZdyM')
    if yt_url != '':
        ytid = get_ytid(yt_url)
        yt_img = f'http://img.youtube.com/vi/{ytid}/sddefault.jpg'
        st.image(yt_img)
        st.write('YouTube video thumbnail image URL: ', yt_img)
    else:
       st.write('☝️ Enter URL to continue ...')