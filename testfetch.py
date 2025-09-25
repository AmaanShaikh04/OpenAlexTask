"""
OpenAlex Research Article Analysis
==================================

Objective:
----------
Explore AI/ML research trends in three journals from 2020 onward. Identify 
articles containing target keywords, summarize publication counts, and visualize trends.

Deliverables:
-------------
1. CSV of all articles matching keywords.
2. Summary table of total vs. keyword-matching articles.
3. Yearly breakdown of publications.
4. Excel report combining all data.
5. Visualizations for trends and keyword share.
"""

# ------------------------------
# Import Libraries
# ------------------------------
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Configuration
# ------------------------------

# Journals dictionary with full OpenAlex URLs
journals = {
    "Scientometrics": "https://openalex.org/s148561398",
    "Quantitative Science Studies": "https://openalex.org/s4210195326",
    "Journal of Informetrics": "https://openalex.org/s205292342"
}

# Keywords to look for (case-insensitive)
keywords = [
    "machine learning", "deep learning", "neural networks", "large language model",
    "LLM", "transformer", "attention mechanism", "transfer learning",
    "reinforcement learning", "RL", "supervised learning", "unsupervised learning",
    "semi-supervised learning", "generative adversarial network", "GAN",
    "autoencoder", "variational autoencoder", "representation learning",
    "self-supervised learning", "Explainable AI", "interpretable machine learning",
    "interpretable AI", "causal inference", "causal discovery", "causal modeling",
    "natural language processing", "natural language understanding",
    "natural language generation", "topic modeling",
    "latent dirichlet allocation", "LDA", "graph neural network", "GNN",
    "embedding", "multimodal learning", "federated learning", "FL",
    "transferability", "zero-shot learning", "few-shot learning", "active learning",
    "human-in-the-loop", "data augmentation", "domain adaptation",
    "knowledge distillation", "prompt tuning", "fine-tuning", "instruction tuning",
    "alignment", "model interpretability", "bias mitigation",
    "algorithmic transparency", "simulation-based inference",
    "probabilistic programming", "Bayesian inference", "variational inference",
    "Bayesian networks", "gibbs sampling", "Markov chain Monte Carlo",
    "Markov decision process", "partially observable Markov decision process",
    "hidden Markov model", "q-learning", "policy gradient",
    "stochastic gradient descent", "logistic regression", "support vector machine",
    "random forest", "k-means", "decision tree", "ensemble learning",
    "dimensionality reduction", "principal component analysis",
    "latent semantic analysis", "robotics", "computer vision",
    "unsupervised clustering", "evolutionary algorithms",
    "particle swarm optimization", "decision support systems",
    "knowledge representation", "object detection", "stochastic optimization",
    "multi-agent systems", "data imputation"
]
keywords = [k.lower() for k in keywords]

# OpenAlex requires a User-Agent header (put your real email)
headers = {"User-Agent": "yourname@example.com"}  # <-- put your email here

# ------------------------------
# Function: Fetch Articles
# ------------------------------

def fetch_journal_articles(journal_id, journal_name, max_pages=10):
    """
    Fetch works from a journal since 2020.

    Args:
        journal_url (str): Full OpenAlex URL of the journal
        journal_name (str): Name for logging
        max_pages (int): Max pages to fetch (200 articles per page)

    Returns:
        list of dicts: Each dict has article metadata and keyword match flag
    """
    url = "https://api.openalex.org/works"
    all_results = []
    page = 1

    while page <= max_pages:
        params = {
            "filter": f"locations.source.id:{journal_id},from_publication_date:2020-01-01",
            "per_page": 200,
            "page": page
        }
        resp = requests.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            print(f"Error fetching {journal_name}, page {page}: {resp.status_code}")
            break

        data = resp.json()
        if "results" not in data:
            break

        for work in data["results"]:
            title = work.get("title", "")
            abstract = work.get("abstract", "")
            pub_date = work.get("publication_date", "")
            year = pub_date.split("-")[0] if pub_date else None
            doi = work.get("doi")
            url_w = work.get("id")

            text = (title + " " + abstract).lower()
            has_keyword = any(kw in text for kw in keywords)

            all_results.append({
                "journal": journal_name,
                "title": title,
                "year": year,
                "doi": doi,
                "url": url_w,
                "has_keyword": has_keyword
            })

        page += 1

    return all_results

# ------------------------------
# Main: Fetch all journals
# ------------------------------
all_articles = []
for jname, jid in journals.items():
    print(f"Fetching {jname}...")
    articles = fetch_journal_articles(jid, jname, max_pages=10)  # adjust max_pages if needed
    all_articles.extend(articles)

# Convert to DataFrame
df = pd.DataFrame(all_articles)

# ------------------------------
# Filter keyword-matching articles and save CSV
# ------------------------------
df_keywords = df[df["has_keyword"] == True]
df_keywords.to_csv("articles_with_keywords.csv", index=False)
print("Saved keyword-matching articles to articles_with_keywords.csv")

# ------------------------------
# Create summary table
# ------------------------------
total_articles = len(df)
keyword_articles = len(df_keywords)
share = (keyword_articles / total_articles * 100) if total_articles > 0 else 0

summary_df = pd.DataFrame({
    "Total Articles": [total_articles],
    "Articles with Keywords": [keyword_articles],
    "Share (%)": [round(share, 2)]
})
summary_df.to_csv("summary_table.csv", index=False)
print("Saved summary table to summary_table.csv")

# ------------------------------
# Yearly breakdown
# ------------------------------
yearly = df.groupby(["year", "has_keyword"]).size().unstack(fill_value=0)
yearly['Share (%)'] = yearly[True] / yearly.sum(axis=1) * 100
yearly = yearly.reset_index()
yearly.to_csv("yearly_breakdown.csv", index=False)
print("Saved yearly breakdown to yearly_breakdown.csv")

# ------------------------------
# Create Excel report with all sheets
# ------------------------------
with pd.ExcelWriter("OpenAlex_Report.xlsx") as writer:
    df_keywords.to_excel(writer, sheet_name="Keyword Articles", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    yearly.to_excel(writer, sheet_name="Yearly Breakdown", index=False)
print("Saved full Excel report to OpenAlex_Report.xlsx")

# ------------------------------
# Visualizations
# ------------------------------
# Stacked bar: keyword vs other articles per year
plt.figure(figsize=(10,6))
plt.bar(yearly['year'], yearly[True], color='skyblue', label='Keyword Articles')
plt.bar(yearly['year'], yearly[False], bottom=yearly[True], color='lightgrey', label='Other Articles')
plt.xlabel('Year')
plt.ylabel('Number of Articles')
plt.title('Yearly Articles with Keyword Highlights')
plt.legend()
plt.tight_layout()
plt.savefig("yearly_articles_bar.png")
plt.show()

# Keyword share line chart
plt.figure(figsize=(10,6))
plt.plot(yearly['year'], yearly['Share (%)'], marker='o', color='green')
plt.xlabel('Year')
plt.ylabel('Keyword Share (%)')
plt.title('Yearly Keyword Share (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig("yearly_keyword_share.png")
plt.show()
