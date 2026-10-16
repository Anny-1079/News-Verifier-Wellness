# News-Verifier-Wellness

# News Verifier & Mental Wellness Assistant

A Streamlit news discovery, summarization, sentiment, and fact-check evidence demo.

## What changed

- Uses NewsAPI when `NEWS_API_KEY` is configured and GDELT as a no-key news discovery source.
- Removes duplicate URLs/headlines and filters weakly relevant results.
- Generates AI summaries only for displayed articles to reduce API usage.
- Falls back to the publisher description when Groq is unavailable.
- Replaces deprecated Matplotlib colormap usage with stable plotting calls.
- Handles empty results, API timeouts, provider errors, and missing API keys.
- Offers optional Google Fact Check Tools API review search.
- Clearly distinguishes sentiment, corroboration, and verified truth.
- Exports displayed results to CSV.

## Important limitations

- GDELT search coverage and relevance vary. No search engine can guarantee complete coverage.
- NewsAPI availability and quotas depend on the account/plan.
- Groq free-tier quotas and model availability can change. If the API is missing or rate-limited, article descriptions are used instead of AI summaries.
- Google Fact Check Tools API requires its own API key and may return no matching reviews.
- Cross-source reporting and sentiment are not proof that a claim is true. This app does not automatically issue a definitive true/false verdict.
- Use only provider APIs in accordance with their terms and current quotas.

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .streamlit_secrets_example.toml .streamlit\secrets.toml
```

Create the `.streamlit` folder first if it does not exist, then edit `.streamlit/secrets.toml` and enter your keys.

```powershell
streamlit run app.py
```

You can run without any API keys: GDELT results and sentiment still work, while summaries use source descriptions. Configure `NEWS_API_KEY` for NewsAPI, `GROQ_API_KEY` for AI summaries, and `GOOGLE_FACT_CHECK_API_KEY` for third-party fact-check reviews.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `.gitignore` to your GitHub repository.
2. Do not upload `.streamlit/secrets.toml` or any API keys to GitHub.
3. In Streamlit Community Cloud, create/deploy the app by selecting the repository and `app.py`.
4. Open the app's settings / Secrets section and add TOML entries, for example:

```toml
NEWS_API_KEY = "your_newsapi_key"
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama-3.1-8b-instant"
GOOGLE_FACT_CHECK_API_KEY = "optional_google_fact_check_api_key"
```

5. Deploy/reboot the app. If a provider returns errors, check the Streamlit app logs and the provider dashboard.

## Suggested testing checklist

- Search a broad topic such as `climate change`.
- Search a specific organization such as `Lovely Professional University`.
- Search a factual claim and check whether the fact-check API returns reviews.
- Try with `GROQ_API_KEY` omitted to verify fallback summaries.
- Try with `NEWS_API_KEY` omitted to verify GDELT fallback.
- Test a nonsense query and confirm the app shows a friendly no-results message.
- Confirm that repeated articles are deduplicated and that charts still render for a small result set.
