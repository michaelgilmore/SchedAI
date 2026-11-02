# SchedAI: ClearLink scheduling prototype

A demo prototype built as a technical interview exercise.
ClearLink Communications is a fictional regional internet provider, and all data
is synthetic; no affiliation with any real company of that name is implied.
ClearLink books every field service job into a 2-hour slot; this app shows
why that misses the 80% on-time target and what smarter scheduling would do.

## Pages
- **Recommend a tech**: enter a job type, zone and day; get the best-fit tech,
  predicted duration, slot to book, and plain-English reasons.
- **Impact**: replays the last 6 months of tickets under the new rules
  (62% on time today, about 90% with right-sized slots and skill-aware assignment).
- **Data analysis**: on-time, delay and rating trends with filters, a day view,
  and job mix over time.

## Run locally
    pip install -r requirements.txt
    streamlit run clearlink_trends.py

Scheduling logic lives in `recommender.py`. The data (`clearlink_service_tickets.csv`)
is a synthetic dataset provided with the exercise.

## Sign-in
The app asks for a username and password. Logins are kept in Streamlit secrets,
not in this repo:
- **Locally:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
  and set passwords. That file is git-ignored.
- **Streamlit Cloud:** paste the same `[users]` section into the app's
  Settings → Secrets.
