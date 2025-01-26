import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title("SMS Spam Detection Model")
st.write("*This is trained model for finding msg is spam or not*")
    

input_sms = st.text_area("Enter the SMS to classify ",height = 150)

if st.button('Predict'):
    if input_sms:
        data = [input_sms]
        vector_input = tk.transform(data).toarray()
        result = model.predict(vector_input)
        if result[0]==0:
            st.header("This is NOT a SPAM Message")
        else:
            st.header("This is a SPAM Message")
    else:
        st.header("Plese enter the SMS to Classify")




    
        