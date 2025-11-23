# ai_news_verifier_app/app.py
import os
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from datetime import datetime, timedelta
from groq import Groq
from textblob import TextBlob
from matplotlib import cm
from time import sleep

# --- API KEYS ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def main():
    
    # --- PAGE CONFIG ---
    st.set_page_config(page_title="News Verifier & Wellness", layout="wide")
    st.markdown("""
    <style>
    body {
        background-color: #242526;
        align: center;
    }
    .news-block {
        background-color: black;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: white;
    }
    .sentiment-positive {
        color: green;
        font-weight: bold;
    }
    .sentiment-negative {
        color: red;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: gray;
        font-weight: bold;
    }         
    </style>
    """, unsafe_allow_html=True)

    st.title("📰 News Verifier & Mental Wellness Assistant")
    st.markdown("""Enter a topic to explore real news, check for misinformation, assess public sentiment 📊, and receive AI-powered mental wellness insights 🧠.
    """)

    # --- USER INPUT ---
    keyword = st.text_input("🔍 Enter a topic (e.g., AI, climate change, women's safety):")
    days = st.slider("📆 Look back over how many days?", min_value=1, max_value=31, value=3)

    analyze = st.button("🔍 Analyze News")

    # --- BUTTON TO TRIGGER ---
    if analyze and keyword:
        # --- FETCH NEWSAPI.ORG ---
        def fetch_newsapi_news(keyword, days):
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            url = f"https://newsapi.org/v2/everything?q={keyword}&from={from_date}&language=en&sortBy=publishedAt&pageSize=30&apiKey={NEWS_API_KEY}"
            res = requests.get(url)
            if res.status_code == 200:
                return res.json().get("articles", [])
            return []

        # --- FETCH POLYGON ---
        def fetch_polygon_news(keyword, days):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            url = f"https://api.polygon.io/v2/reference/news?ticker={keyword.upper()}&published_utc.gte={start_date.strftime('%Y-%m-%d')}&sort=published_utc&order=desc&limit=50&apiKey={POLYGON_API_KEY}"
            res = requests.get(url)
            if res.status_code == 200:
                return res.json().get("results", [])
            return []

        # --- ANALYZE SENTIMENT ---
        def analyze_sentiment(text):
            return round(TextBlob(text).sentiment.polarity, 7)

        # --- GROQ FAKE NEWS & WELLNESS ---
        def llm_fake_check(news_list):
            prompt = f"""
    You are an AI truth checker and wellness advisor. Analyze the following news headlines.
    - Detect any that seem fake or manipulative.
    - Summarize the overall trend.
    - Provide mental wellness tips if sentiment is negative.

    News:
    {news_list}
    """
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    # "llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"⚠️ Groq API error: {str(e)}")
                return "(Unable to generate summary due to rate limit or API error.)"

        newsapi_news = fetch_newsapi_news(keyword, days)[:15]
        polygon_news = fetch_polygon_news(keyword, days)[:15]  # get first 3 from Polygon

        headlines = []
        sentiments = []
        summaries = []
        links = []

        for item in newsapi_news:
            title = item.get("title", "")
            desc = item.get("description", "")
            url = item.get("url", "")
            full = title + " " + desc
            sentiment = analyze_sentiment(full)
            sentiment_label = "Positive" if sentiment > 0.2 else "Negative" if sentiment < -0.2 else "Neutral"

            try:
                prompt = f"Summarize this news in one short paragraph: {full}"
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.choices[0].message.content.strip()
            except Exception as e:
                summary = "(Summary not available due to rate limit.)"

            headlines.append(title)
            sentiments.append(sentiment)
            summaries.append(summary)
            links.append(url)

        for item in polygon_news:
            title = item.get("title", "")
            desc = item.get("description", "")
            url = item.get("article_url", "")
            full = title + " " + desc
            sentiment = analyze_sentiment(full)
            sentiment_label = "Positive" if sentiment > 0.2 else "Negative" if sentiment < -0.2 else "Neutral"

            try:
                prompt = f"Summarize this news in one short paragraph: {full}"
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.choices[0].message.content.strip()
            except Exception as e:
                summary = "(Summary not available due to rate limit.)"

            headlines.append(title)
            sentiments.append(sentiment)
            summaries.append(summary)
            links.append(url)

        # --- DISPLAY NEWS ---
        st.subheader("📰 News Analysis")
        for i in range(min(7, len(headlines))):  # only display first 7
            sentiment_label = "Positive" if sentiments[i] > 0.2 else "Negative" if sentiments[i] < -0.2 else "Neutral"
            sentiment_class = "sentiment-positive" if sentiment_label == "Positive" else "sentiment-negative" if sentiment_label == "Negative" else "sentiment-neutral"
            with st.container():
                st.markdown(f"<div class='news-block'>"
                            f"<b>{headlines[i]}</b><br>"
                            f"<span class='{sentiment_class}'>Sentiment: {sentiment_label}</span><br><br>"
                            f"{summaries[i]}<br><br>"
                            f"<a href='{links[i]}' target='_blank'>Read Full News</a>"
                            f"</div>", unsafe_allow_html=True)


        # --- CHARTS SIDE BY SIDE ---
        st.subheader("📊 Sentiment Charts")
        st.caption("📈 Sentiment based on top 30 news articles across sources (only 7 shown above).")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#111111')
            ax.set_facecolor('#111111')
            cmap = cm.get_cmap('RdYlGn')
            colors = [cmap((s + 1) / 2) for s in sentiments]
            ax.set_title("Sentiment Polarity", color='white', pad=20)
            ax.bar(range(len(sentiments)), sentiments, color=colors)
            ax.set_xlabel("Article Index", color='white')
            ax.set_ylabel("Polarity (-1 to 1)", color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            st.pyplot(fig)

        with col2:
            positive = sum(1 for s in sentiments if s > 0.2)
            neutral = sum(1 for s in sentiments if -0.2 <= s <= 0.2)
            negative = sum(1 for s in sentiments if s < -0.2)
            pie_fig, pie_ax = plt.subplots(figsize=(5, 4))
            pie_fig.patch.set_facecolor('#111111')
            pie_ax.set_facecolor('#111111')
            pie_ax.pie([positive, neutral, negative], labels=["Positive", "Neutral", "Negative"],
                    colors=["#28a745", "#6c757d", "#dc3545"], autopct='%1.1f%%', textprops={'color':"white"})
            pie_ax.set_title("Overall Sentiment Distribution", color='white')
            st.pyplot(pie_fig)


        # --- GROQ SUMMARY ---
        st.subheader("🧠 AI Fake News Check & Wellness Advice")
        combined = "\n".join(headlines)
        ai_response = llm_fake_check(combined)
        if ai_response:
            ai_response = ai_response.replace("**Analysis:**", "").strip()
        st.markdown(f"<div class='news-block'>{ai_response}</div>", unsafe_allow_html=True)

    elif keyword:
        st.info("Press the 'Analyze News' button to see results.")

if __name__ == "__main__":
    main()
