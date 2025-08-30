import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
# Download stopwords
nltk.download("stopwords")
stop_words=set(stopwords.words("english"))
ps=PorterStemmer()
# Text Preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text=text.lower()
    text=re.sub(r"[^a-z\s]", "", text)
    words=text.split()
    words=[w for w in words if w not in stop_words]
    words=[ps.stem(w) for w in words]
    return " ".join(words)
# Load the Data
@st.cache_data
def load_data():
    df=pd.read_excel(r"C:\Users\nirai\Downloads\chatgpt_style_reviews_dataset.xlsx")
# Create clean_text column
    if "review" in df.columns:
        df["clean_text"]=df["review"].apply(preprocess_text)
    elif "text_raw" in df.columns:
        df["clean_text"]=df["text_raw"].apply(preprocess_text)
    else:
        st.error("No review/text column found in dataset!")
        return df
    df["review_length"]=df["clean_text"].apply(lambda x: len(str(x).split()))
    return df
data = load_data()
# Sentiment Mapping
def map_sentiment(rating):
    if rating>=4:
        return "Positive"
    elif rating==3:
        return "Neutral"
    else:
        return "Negative"
if "rating" in data.columns:
    data["sentiment"]=data["rating"].apply(map_sentiment)
st.title("ChatGPT User Reviews - Sentiment Analysis")
# Overall sentiment
st.header("Overall Sentiment Distribution")
sent_dist=data["sentiment"].value_counts(normalize=True)
st.bar_chart(sent_dist)
# Sentiment vs Rating
st.header("Sentiment vs Rating")
if "rating" in data.columns:
    fig,ax=plt.subplots()
    sns.countplot(x="rating",hue="sentiment",data=data,ax=ax)
    st.pyplot(fig)
# Keywords per sentiment
st.header("Keywords per Sentiment Class")
for sentiment in ["Positive","Neutral","Negative"]:
    text=" ".join(data.loc[data["sentiment"]==sentiment,"clean_text"])
    if text.strip():
        wc=WordCloud(width=600,height=400,background_color="white").generate(text)
        st.subheader(f"{sentiment} Reviews WordCloud")
        st.image(wc.to_array())
# Sentiment over time
st.header("Sentiment Trend Over Time")
if "date" in data.columns:
    data["date"]=pd.to_datetime(data["date"],errors="coerce")
    trend=data.groupby([pd.Grouper(key="date",freq="M"),"sentiment"]).size().unstack()
    st.line_chart(trend)
# Verified vs Non-Verified
st.header("Verified Users Sentiment Distribution")
if "verified_purchase" in data.columns:
    fig,ax=plt.subplots(figsize=(6,4))
    sns.countplot(x="verified_purchase",hue="sentiment",data=data,ax=ax)
    ax.set_title("Sentiment Distribution: Verified vs Non-Verified")
    st.pyplot(fig)
    st.write(data.groupby("verified_purchase")["sentiment"].value_counts(normalize=True).unstack())
else:
    st.warning("Column 'verified_purchase' not found in dataset.")
# Review length vs Sentiment
st.header("Review Length vs Sentiment")
fig,ax=plt.subplots()
sns.boxplot(x="sentiment",y="review_length",data=data,ax=ax)
st.pyplot(fig)
# Sentiment by Location
st.header("Sentiment by Location")
if "location" in data.columns:
    loc_sent=data.groupby("location")["sentiment"].value_counts(normalize=True).unstack().fillna(0)
    st.dataframe(loc_sent)
# Sentiment by Platform
st.header("Sentiment by Platform (Web vs Mobile)")
if "platform" in data.columns:
    fig,ax=plt.subplots()
    sns.countplot(x="platform",hue="sentiment",data=data,ax=ax)
    st.pyplot(fig)
# Sentiment by ChatGPT Version
st.header("Sentiment by ChatGPT Version")
if "version" in data.columns:
    ver_sent=data.groupby("version")["sentiment"].value_counts(normalize=True).unstack().fillna(0)
    st.bar_chart(ver_sent)
# Common negative feedback themes
st.header("Most Common Negative Feedback Themes")
neg_text=" ".join(data.loc[data["sentiment"]=="Negative","clean_text"])
if neg_text.strip():
    neg_wc=WordCloud(width=800,height=400,background_color="white").generate(neg_text)
    st.image(neg_wc.to_array())