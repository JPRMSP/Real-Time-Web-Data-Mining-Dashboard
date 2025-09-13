import streamlit as st
import requests
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import networkx as nx

# ---------------------------
# Utility Functions
# ---------------------------
def fetch_text_and_links(url):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style"]):
            s.extract()
        text = " ".join(soup.stripped_strings)
        # Extract outgoing links
        links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].startswith("http")]
        return text, links
    except Exception as e:
        return f"Error fetching URL: {e}", []

def preprocess(text):
    text = text.lower()
    words = re.findall(r"\b[a-z]{3,}\b", text)

    # Fetch stopwords dynamically from GitHub
    stopwords_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt"
    stopwords_text = requests.get(stopwords_url).text
    stopwords = set(stopwords_text.split())

    words = [w for w in words if w not in stopwords]
    return words

def word_frequencies(words, top_n=20):
    return Counter(words).most_common(top_n)

def co_occurrence(words, window=2):
    pairs = defaultdict(int)
    for i in range(len(words) - window):
        pair = tuple(sorted(words[i:i+window]))
        pairs[pair] += 1
    return sorted(pairs.items(), key=lambda x: -x[1])[:10]

def sentiment_analysis(words):
    emotions = {
        "Joy": {"happy","joy","love","great","excellent","success","win"},
        "Sadness": {"sad","loss","fail","cry"},
        "Anger": {"angry","hate","fight","bad","terrible"},
        "Fear": {"fear","worry","danger","risk"},
        "Trust": {"trust","safe","secure","reliable"},
    }
    counts = {emo: sum(w in lex for w in words) for emo, lex in emotions.items()}
    return counts

def build_link_graph(results):
    G = nx.DiGraph()
    for url, data in results.items():
        G.add_node(url)
        for link in data["links"][:10]:  # limit to avoid clutter
            G.add_edge(url, link)
    return G

def compute_pagerank(G):
    return nx.pagerank(G)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Real-Time Web Data Mining Dashboard", layout="wide")

st.title("🌐 Real-Time Web Data Mining Dashboard")
st.markdown("Analyze live websites with **keyword mining, co-occurrence rules, sentiment/emotion mining, and PageRank link analysis**. All without datasets or pretrained models!")

urls = st.text_area("Enter one or more URLs (comma-separated):", 
                    "https://en.wikipedia.org/wiki/Artificial_intelligence, https://en.wikipedia.org/wiki/Data_mining")

if st.button("Mine Now"):
    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    results = {}

    with st.spinner("Fetching and analyzing websites..."):
        for url in url_list:
            raw_text, links = fetch_text_and_links(url)
            words = preprocess(raw_text)
            if len(words) > 0:
                results[url] = {
                    "text": raw_text,
                    "words": words,
                    "freq": word_frequencies(words, 20),
                    "cooc": co_occurrence(words),
                    "sentiment": sentiment_analysis(words),
                    "links": links
                }

    if not results:
        st.error("No valid data mined from given URLs.")
    else:
        # Tabs for each site
        for url, data in results.items():
            st.header(f"🔎 Analysis for {url}")

            # Frequency
            st.subheader("📊 Top Keywords")
            st.table(data["freq"])

            # Wordcloud
            st.subheader("☁ Word Cloud")
            wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(data["words"]))
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # Co-occurrence
            st.subheader("🔗 Word Co-occurrence")
            st.table(data["cooc"])

            # Sentiment
            st.subheader("😊 Emotion-Based Opinion Mining")
            st.bar_chart(data["sentiment"])

            # Raw Preview
            with st.expander("📝 Raw Text Preview"):
                st.write(data["text"][:2000] + "...")

            # Links found
            with st.expander("🌍 Outgoing Links Found"):
                st.write(data["links"])

        # Cross-site comparison
        if len(results) > 1:
            st.header("📈 Cross-Site Keyword Comparison")
            all_freqs = []
            for url, data in results.items():
                for word, count in data["freq"]:
                    all_freqs.append({"URL": url, "Word": word, "Count": count})
            df = pd.DataFrame(all_freqs)
            pivot_df = df.pivot_table(index="Word", columns="URL", values="Count", fill_value=0)
            st.dataframe(pivot_df)

            st.subheader("Common High-Frequency Words Across Sites")
            common = set.intersection(*[set([w for w,_ in data["freq"]]) for data in results.values()])
            st.write(list(common))

        # ---------------------------
        # Link Graph & PageRank
        # ---------------------------
        st.header("🌐 Link Graph & PageRank")
        G = build_link_graph(results)
        if len(G.nodes) > 0:
            pr = compute_pagerank(G)

            st.subheader("PageRank Scores")
            pr_df = pd.DataFrame(pr.items(), columns=["Page", "PageRank"]).sort_values("PageRank", ascending=False)
            st.table(pr_df.head(10))

            # Graph Visualization
            st.subheader("Graph Visualization")
            fig, ax = plt.subplots(figsize=(10,6))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            nx.draw(G, pos, with_labels=False, node_size=500, node_color="skyblue", arrowsize=12, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
            st.pyplot(fig)

        # Export option
        if len(results) > 1:
            st.download_button("⬇ Export Keyword Data (CSV)", df.to_csv().encode("utf-8"), "web_mining_results.csv", "text/csv")
