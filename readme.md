# OpenAlex Research Article Analysis

## Overview
This project fetches research articles from three journals:

- Scientometrics
- Quantitative Science Studies
- Journal of Informetrics

It filters articles based on a curated list of AI/ML keywords and produces:

1. CSV of keyword-matching articles
2. Summary table (total articles, keyword hits, share %)
3. Yearly breakdown of keyword vs. non-keyword articles
4. Excel report combining all tables in separate sheets
5. Visualizations showing yearly article counts and keyword share

This project demonstrates data fetching, processing, and analysis using Python with a focus on reproducibility and clarity.

---

## Folder Structure

```
OpenAlexTask/
│
├─ testfetch.py                 # Main Python script
├─ articles_with_keywords.csv   # Keyword-matching articles (generated)
├─ summary_table.csv            # Summary table (generated)
├─ yearly_breakdown.csv         # Yearly breakdown (generated)
├─ OpenAlex_Report.xlsx         # Excel report with all sheets (generated)
├─ yearly_articles_bar.png      # Stacked bar chart of yearly articles (generated)
├─ yearly_keyword_share.png     # Line chart of yearly keyword share (generated)
├─ requirements.txt             # Python dependencies
└─ README.md                    # This file
```

---

## Setup Instructions

1. Clone or download this folder to your local machine.

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Script

1. Ensure your virtual environment is activated.
2. Run the Python script:
```bash
python3 testfetch.py
```

3. The script will:
   - Fetch articles from the three journals
   - Filter articles using the keyword list
   - Save CSV files:
     - articles_with_keywords.csv
     - summary_table.csv
     - yearly_breakdown.csv
   - Save an Excel report: OpenAlex_Report.xlsx
   - Generate visualizations:
     - yearly_articles_bar.png
     - yearly_keyword_share.png

---

## Notes

- User-Agent: OpenAlex requires a proper user agent. Replace the placeholder in testfetch.py:
```python
headers = {"User-Agent": "yourname@example.com"}
```

- The script limits results per journal to 10 pages (200 articles per page). You can adjust max_pages in fetch_journal_articles() if needed.

- CSV and Excel outputs are ready for analysis or reporting.

---

## Optional Enhancements

- Increase max_pages to fetch more articles.
- Expand the keyword list for more comprehensive coverage.
- Use visualizations to highlight trends in research publications.
- Add word cloud or top keywords analysis for extra insights.

---


## Acknowledgements

- [OpenAlex API](https://docs.openalex.org/) – Data source for research publications
- Python libraries: pandas, requests, matplotlib, openpyxl, seaborn
