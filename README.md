# SchedAI: ClearLink scheduling prototype

Prototype for the Opmed Forward Deployed Engineer home assignment.
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
is the fictional dataset provided with the assignment.
