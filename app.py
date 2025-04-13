import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import warnings

warnings.filterwarnings("ignore")

# Streamlit config
st.set_page_config(page_title="Sentiment Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("clean_sentimentdataset.csv")

df = load_data()

# Sidebar Filters
with st.sidebar:
    st.title("🧭 Navigation & Filters")
    selected_page = st.radio("Select a Page", ["📊 Dashboard", "📈 Visualizations", "🌍 Geo Map", "📝 Keywords & WordCloud"])

    st.markdown("---")
    st.header("🔍 Filters")
    sentiment_options = ['All'] + sorted(df['Sentiment'].dropna().unique())
    selected_sentiment = st.selectbox("Sentiment", sentiment_options)

    year_options = ['All'] + sorted(df['Year'].dropna().astype(str).unique())
    selected_year = st.selectbox("Year", year_options)

    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_options = ['All'] + month_order
    selected_month = st.selectbox("Month", month_options)

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_options = ['All'] + day_order
    selected_day = st.selectbox("Day of Week", day_options)

# Apply Filters
filtered_df = df.copy()
if selected_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]
if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['Year'].astype(str) == selected_year]
if selected_month != 'All':
    filtered_df = filtered_df[filtered_df['Month'] == selected_month]
if selected_day != 'All':
    filtered_df = filtered_df[filtered_df['Day_of_Week'] == selected_day]

# Page: Dashboard KPIs & Summary
if selected_page == "📊 Dashboard":
    st.title("📊 Sentiment Analysis Dashboard")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="🗃️ Total Records", value=len(filtered_df))
    col2.metric(label="👍 Positive", value=len(filtered_df[filtered_df['Sentiment'] == 'positive']))
    col3.metric(label="😐 Neutral", value=len(filtered_df[filtered_df['Sentiment'] == 'neutral']))
    col4.metric(label="👎 Negative", value=len(filtered_df[filtered_df['Sentiment'] == 'negative']))

    st.markdown("---")

    # Pie Chart
    sentiment_counts = filtered_df['Sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'], fill_value=0)
    fig1 = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, hole=0.6, 
                  title="Sentiment Distribution", color=sentiment_counts.index,
                  color_discrete_map={"positive": "green", "neutral": "blue", "negative": "red"})
    st.plotly_chart(fig1, use_container_width=True)

    # Horizontal Bar Chart: Top Platforms by Total Reviews
    if 'Platform' in filtered_df.columns:
        st.markdown("---")
        st.subheader("🌐 Top Platforms by Total Reviews")

        platform_counts = filtered_df['Platform'].value_counts().reset_index()
        platform_counts.columns = ['Platform', 'Count']

        fig_platform = px.bar(
            platform_counts,
            x='Count',
            y='Platform',
            orientation='h',
            title='Top Platforms by Review Volume',
            color='Count',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_platform, use_container_width=True)


# Page: Visualizations
elif selected_page == "📈 Visualizations":
    st.title("📈 Sentiment Trends Over Time")
    if 'Year' in filtered_df.columns:
        fig_year = px.histogram(filtered_df, x='Year', color='Sentiment', barmode='group',
                                color_discrete_map={"positive": "green", "neutral": "blue", "negative": "red"})
        st.plotly_chart(fig_year, use_container_width=True)

    if 'Month' in filtered_df.columns:
        fig_month = px.histogram(filtered_df, x='Month', color='Sentiment', barmode='group',
                                 category_orders={"Month": month_order},
                                 color_discrete_map={"positive": "green", "neutral": "blue", "negative": "red"})
        st.plotly_chart(fig_month, use_container_width=True)

    if 'Day_of_Week' in filtered_df.columns:
        fig_day = px.histogram(filtered_df, x='Day_of_Week', color='Sentiment', barmode='group',
                               category_orders={"Day_of_Week": day_order},
                               color_discrete_map={"positive": "green", "neutral": "blue", "negative": "red"})
        st.plotly_chart(fig_day, use_container_width=True)

# Page: Geo Map
elif selected_page == "🌍 Geo Map":
    st.title("🌍 Geographic Sentiment Distribution")
    if 'Country' in filtered_df.columns:
        st.markdown("**Sentiment Color Legend**: 🟩 Positive | 🟦 Neutral | 🟥 Negative")

        country_sentiment_df = filtered_df.groupby(['Country', 'Sentiment']).size().reset_index(name='Count')

        fig_map = px.choropleth(country_sentiment_df,
                                locations='Country',
                                locationmode='country names',
                                color='Count',
                                hover_name='Country',
                                animation_frame='Sentiment',
                                title='Country-wise Mentions per Sentiment',
                                color_continuous_scale='Viridis')
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("No 'Country' column available for map visualization.")

# Page: Text & Keywords
elif selected_page == "📝 Keywords & WordCloud":
    st.title("📝 Textual Data Insights")

    # Top Keywords
    def display_top_keywords(sentiment, data, n=10):
        text_data = data[data['Sentiment'] == sentiment]['Text'].dropna().astype(str)
        if not text_data.empty:
            vec = CountVectorizer(stop_words='english', max_features=1000)
            X = vec.fit_transform(text_data)
            sum_words = X.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            top_words = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
            df_words = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            fig = px.bar(df_words, x='Word', y='Frequency', title=f"Top {n} Keywords - {sentiment.capitalize()}",
                         color_discrete_sequence=["green" if sentiment == "positive" else 
                                                  "blue" if sentiment == "neutral" else "red"])
            st.plotly_chart(fig, use_container_width=True)

    if selected_sentiment != 'All':
        display_top_keywords(selected_sentiment, filtered_df)
    else:
        for sent in ['positive', 'neutral', 'negative']:
            display_top_keywords(sent, filtered_df)

    # WordCloud
    st.subheader("☁️ Word Cloud")
    def generate_wordcloud(sentiment, data):
        text = " ".join(data[data['Sentiment'] == sentiment]['Text'].dropna().astype(str))
        if text:
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info(f"No text data for {sentiment} sentiment.")

    if selected_sentiment != 'All':
        generate_wordcloud(selected_sentiment, filtered_df)
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Positive**")
            generate_wordcloud('positive', filtered_df)
        with col2:
            st.markdown("**Neutral**")
            generate_wordcloud('neutral', filtered_df)
        with col3:
            st.markdown("**Negative**")
            generate_wordcloud('negative', filtered_df)
