"""
ClearLink scheduling prototype: recommend a tech, show the impact, explore the data.

Run:
    pip install -r requirements.txt     (streamlit>=1.36, pandas, plotly)
    streamlit run clearlink_trends.py

Needs clearlink_service_tickets.csv and recommender.py in the same folder.
The Data analysis page can also take an uploaded CSV from the sidebar.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import recommender as rec

DEFAULT_CSV = Path(__file__).parent / "clearlink_service_tickets.csv"
FILTER_COLS = ["tech_seniority", "zone", "job_type"]
PERIODS = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}

DAY_START_HOUR = 8
SLOT_HOURS = 2
SLOTS_PER_DAY = 4
SLOT_PAD = 0.08  # hours trimmed from each end of a dashed slot (~5 minutes)
LUNCH_START, LUNCH_END = 12, 13
JOB_COLORS = ["#1D4ED8", "#C2410C", "#15803D", "#7C3AED", "#B45309", "#0E7490", "#BE185D"]

@st.cache_data
def load_data(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["delay_minutes", "customer_rating", "actual_duration_min"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # on_time_completion may arrive as True/False, "yes"/"no", or 1/0
    df["on_time_completion"] = (
        df["on_time_completion"].astype(str).str.strip().str.lower()
        .isin(["1", "true", "yes", "y", "t"])
    )
    for col in FILTER_COLS:
        df[col] = df[col].astype(str)
    return df


def summarize(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate tickets into one row per period."""
    return (
        df.set_index("date")
        .resample(freq)
        .agg(
            tickets=("ticket_id", "count"),
            avg_delay=("delay_minutes", "mean"),
            avg_rating=("customer_rating", "mean"),
            on_time_rate=("on_time_completion", "mean"),
        )
        .dropna(subset=["avg_delay"])
        .reset_index()
    )


def duration_table(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Average actual duration per combination of the chosen columns."""
    table = (
        df.groupby(group_cols)
        .agg(
            tickets=("ticket_id", "count"),
            avg_actual_min=("actual_duration_min", "mean"),
            avg_scheduled_min=("scheduled_duration_min", "mean"),
            avg_rating=("customer_rating", "mean"),
        )
        .reset_index()
    )
    table["avg_over_slot_min"] = table["avg_actual_min"] - table["avg_scheduled_min"]
    table = table[group_cols + ["tickets", "avg_actual_min", "avg_over_slot_min", "avg_rating"]]
    return (
        table.round({"avg_actual_min": 1, "avg_over_slot_min": 1, "avg_rating": 2})
        .sort_values("avg_actual_min", ascending=False)
    )
    

def day_slots(day_start=DAY_START_HOUR):
    """Booked slots for a full day, skipping lunch. 8 AM start gives
    8–10, 10–12, (lunch 12–1), 1–3, 3–5."""
    slots, t = [], float(day_start)
    while len(slots) < SLOTS_PER_DAY:
        if t < LUNCH_END and t + SLOT_HOURS > LUNCH_START:  # slot would overlap lunch
            t = LUNCH_END
        slots.append((t, t + SLOT_HOURS))
        t += SLOT_HOURS
    return slots
    

def day_timeline(day_df: pd.DataFrame, day_start: float = DAY_START_HOUR,
                 protect_lunch: bool = False) -> pd.DataFrame:
    """Rebuild one day's schedule per tech (hours as decimals, e.g. 9.5 = 9:30).

    Job k for a tech is booked into the k-th slot of the day template (lunch skipped).
    A job starts at its slot or when the tech's previous job ends, whichever is later.
    protect_lunch=True: a job that would start during lunch waits until lunch ends.
    """
    slots = day_slots(day_start)
    out = []
    for _, jobs in day_df.sort_values("ticket_id").groupby("tech_id"):
        prev_end = float(day_start)
        for k, (_, j) in enumerate(jobs.iterrows()):
            if k < len(slots):
                slot_start, slot_end = slots[k]
            else:  # more jobs than slots: keep booking back to back after the last one
                slot_start = slots[-1][1] + (k - len(slots)) * SLOT_HOURS
                slot_end = slot_start + SLOT_HOURS

            start = max(slot_start, prev_end)
            if protect_lunch and LUNCH_START <= start < LUNCH_END:
                start = LUNCH_END
            end = start + j["actual_duration_min"] / 60

            out.append({**j.to_dict(), "slot_start": slot_start, "slot_end": slot_end,
                        "act_start": start, "act_end": end})
            prev_end = end
    return pd.DataFrame(out)


def clock(h: float) -> str:
    total = int(round(h * 60))
    hh, mm = divmod(total, 60)
    return f"{(hh - 1) % 12 + 1}:{mm:02d} {'AM' if hh < 12 else 'PM'}"


def make_day_chart(tl: pd.DataFrame, day_start: float) -> go.Figure:
    techs = sorted(tl["tech_id"].unique(), key=lambda t: (len(str(t)), str(t)))
    xpos = {t: i for i, t in enumerate(techs)}
    tl = tl.assign(x=tl["tech_id"].map(xpos))
    types = sorted(tl["job_type"].unique())
    color = {t: JOB_COLORS[i % len(JOB_COLORS)] for i, t in enumerate(types)}

    fig = go.Figure()
    # Booked slots: dashed outlines behind the actual work.
    for i in range(len(techs)):
        for s0, s1 in day_slots(day_start):
            fig.add_shape(type="rect", x0=i - 0.42, x1=i + 0.42, y0=s0 + SLOT_PAD, y1=s1 - SLOT_PAD,
                          line=dict(color="#9CA3AF", width=1, dash="dash"),
                          fillcolor="rgba(0,0,0,0)", layer="below")    

    fig.add_hrect(y0=LUNCH_START, y1=LUNCH_END, fillcolor="#9CA3AF", opacity=0.15,
              line_width=0, layer="below",
              annotation_text="Lunch", annotation_position="left")                          
                          
    # Actual work: one trace per job type so the legend explains the colors.
    for jt in types:
        sub = tl[tl["job_type"] == jt]
        late = sub["act_end"] > sub["slot_end"]
        fig.add_trace(go.Bar(
            x=sub["x"], base=sub["act_start"], y=sub["act_end"] - sub["act_start"],
            width=0.6, name=jt, marker=dict(color=color[jt], opacity=0.85,
                                            line=dict(color=["#DC2626" if l else color[jt]
                                                             for l in late],
                                                      width=[3 if l else 1 for l in late])),
            customdata=list(zip(sub["ticket_id"], sub["act_start"].map(clock),
                                sub["act_end"].map(clock), sub["slot_start"].map(clock),
                                sub["slot_end"].map(clock), sub["actual_duration_min"],
                                sub["tech_seniority"], sub["zone"],
                                sub["delay_minutes"], sub["customer_rating"])),
            hovertemplate=("<b>Ticket %{customdata[0]}</b> · " + str(jt) +
                           "<br>Worked %{customdata[1]}–%{customdata[2]} "
                           "(%{customdata[5]} min)<br>Booked %{customdata[3]}–%{customdata[4]}"
                           "<br>%{customdata[6]} tech, zone %{customdata[7]}"
                           "<br>Recorded delay %{customdata[8]} min, rating %{customdata[9]}"
                           "<extra></extra>"),
        ))

    last = max(tl["act_end"].max(), day_slots(day_start)[-1][1])
    
    hours = list(range(int(day_start), int(last) + 2))
    
    height = max(500, int((last - day_start) * 70) + 190)
    TOP = 150  # px reserved above the plot: title + legend + tech labels + axis title

    fig.update_layout(
        barmode="overlay", height=height,
        title=dict(text="Booked slots (dashed) vs. actual work (solid); red outline = ran past its slot",
                   yref="container", y=1 - 12 / height, yanchor="top"),
        legend=dict(title="Job type", orientation="h", x=0,
                    yref="container", y=1 - 48 / height, yanchor="top"),
        xaxis=dict(tickvals=list(range(len(techs))), ticktext=[f"Tech {t}" for t in techs],
                   title="Technician", range=[-0.6, len(techs) - 0.4], side="top"),
        yaxis=dict(autorange="reversed", tickvals=hours, ticktext=[clock(h) for h in hours],
                   title="Time of day", gridcolor="#E5E7EB"),
        margin=dict(t=TOP, b=30),
    )
    return fig


TARGET_ON_TIME = 0.80


def make_job_mix_chart(df: pd.DataFrame, freq: str, period_name: str) -> go.Figure:
    """Number of jobs per job type in each period, one line per job type."""
    counts = (
        df.set_index("date")
        .groupby("job_type")
        .resample(freq)["ticket_id"].count()
        .rename("jobs")
        .reset_index()
    )
    types = sorted(counts["job_type"].unique())
    fig = go.Figure()
    for i, jt in enumerate(types):
        sub = counts[counts["job_type"] == jt]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["jobs"], mode="lines+markers", name=jt,
            line=dict(color=JOB_COLORS[i % len(JOB_COLORS)]),
            hovertemplate="%{x|%b %d, %Y}<br>" + jt + ": %{y} jobs<extra></extra>",
        ))
    fig.update_layout(
        height=420, hovermode="x unified",
        yaxis=dict(title="Jobs per " + {"Daily": "day", "Weekly": "week",
                                         "Monthly": "month"}.get(period_name, "period"),
                   rangemode="tozero"),
        legend=dict(title="Job type", orientation="h", y=-0.15, x=0, yanchor="top"),
        margin=dict(t=20, b=80),
    )
    return fig


def make_on_time_chart(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=summary["date"], y=summary["on_time_rate"], mode="lines+markers",
                   name="On-time rate", line=dict(color="#15803D"),
                   customdata=summary[["tickets"]],
                   hovertemplate="%{x|%b %d, %Y}<br>%{y:.1%} on time<br>"
                                 "%{customdata[0]} tickets<extra></extra>")
    )
    fig.add_hline(y=TARGET_ON_TIME, line_dash="dash", line_color="#6B7280",
                  annotation_text="80% target", annotation_position="top left")
    fig.update_layout(title="On-time completion rate", height=340,
                      yaxis=dict(tickformat=".0%", range=[0, 1]),
                      showlegend=False, margin=dict(t=50, b=20))
    return fig


def make_chart(summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Average delay (minutes)", "Average customer rating"),
    )
    fig.add_trace(
        go.Scatter(x=summary["date"], y=summary["avg_delay"], mode="lines+markers",
                   name="Avg delay (min)", line=dict(color="#C2410C"),
                   customdata=summary[["tickets"]],
                   hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f} min<br>%{customdata[0]} tickets<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=summary["date"], y=summary["avg_rating"], mode="lines+markers",
                   name="Avg rating", line=dict(color="#1D4ED8"),
                   hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}<extra></extra>"),
        row=2, col=1,
    )
    fig.update_layout(height=620, showlegend=False, margin=dict(t=40, b=20),
                      hovermode="x unified")
    return fig


def main():
    st.title("ClearLink: data analysis")

    st.sidebar.header("Data")
    upload = st.sidebar.file_uploader("Service ticket CSV", type="csv")
    if upload is not None:
        df = load_data(upload)
    elif DEFAULT_CSV.exists():
        df = load_data(DEFAULT_CSV)
    else:
        st.info("Upload clearlink_service_tickets.csv in the sidebar, or put it next to this script.")
        return

    st.sidebar.header("Filters")
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    start = st.sidebar.date_input("Start date", min_date,
                                  min_value=min_date, max_value=max_date)
    end = st.sidebar.date_input("End date", max_date,
                                min_value=min_date, max_value=max_date)
    if start > end:
        st.sidebar.error("Start date must be on or before end date.")
        return
    filtered = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]

    for col in FILTER_COLS:
        options = sorted(df[col].unique())
        picked = st.sidebar.multiselect(col.replace("_", " ").title(), options,
                                        placeholder="All")
        if picked:
            filtered = filtered[filtered[col].isin(picked)]

    period = st.sidebar.radio("Group by", list(PERIODS), index=2)

    if filtered.empty:
        st.warning("No tickets match these filters.")
        return

    summary = summarize(filtered, PERIODS[period])
    st.plotly_chart(make_on_time_chart(summary), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets", f"{len(filtered):,}")
    c2.metric("On-time rate", f"{filtered['on_time_completion'].mean():.1%}")
    c3.metric("Avg delay", f"{filtered['delay_minutes'].mean():.1f} min")
    c4.metric("Avg rating", f"{filtered['customer_rating'].mean():.2f}")

    st.plotly_chart(make_chart(summary), use_container_width=True)

    with st.expander("Data table"):
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Average actual duration")
    group_cols = st.multiselect("Break down by", FILTER_COLS, default=FILTER_COLS,
                                format_func=lambda c: c.replace("_", " ").title())
    if group_cols:
        st.dataframe(duration_table(filtered, group_cols),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("Pick at least one column to break down by.")

    st.subheader("Day view")
    st.caption(
        "The data has dates but no clock times, so each tech's day is rebuilt: jobs run "
        "in ticket order, booked into back-to-back 2-hour slots from the day start, and a "
        "job can't start until the tech's previous job ends. Dashed boxes are the booked "
        "slots; solid boxes are when the work actually happened."
    )
    days = sorted(filtered["date"].dt.date.unique())
    dc1, dc2 = st.columns([2, 1])
    day = dc1.date_input("Day", days[-1], min_value=days[0], max_value=days[-1])
    day_start = dc2.number_input("Day starts at (hour)", 5, 12, DAY_START_HOUR)

    # Rebuild from ALL of that day's tickets so the cascade is right, then apply filters.
    timeline = day_timeline(df[df["date"].dt.date == day], day_start)
    timeline = timeline[timeline["ticket_id"].isin(filtered["ticket_id"])]
    if timeline.empty:
        st.info("No jobs on this day match the filters.")
    else:
        late = (timeline["act_end"] > timeline["slot_end"]).sum()
        st.markdown(f"**{len(timeline)} jobs, {timeline['tech_id'].nunique()} techs, "
                    f"{late} finished after their booked slot.**")
        st.plotly_chart(make_day_chart(timeline, day_start), use_container_width=True)

    st.subheader("Jobs by job type over time")
    st.plotly_chart(make_job_mix_chart(filtered, PERIODS[period], period),
                    use_container_width=True)


# ---------------------------------------------------------------------------
# Recommend and Impact pages
# ---------------------------------------------------------------------------

@st.cache_data
def cached_stats(df: pd.DataFrame):
    return rec.build_stats(df)


@st.cache_data
def cached_impact(df: pd.DataFrame):
    res = rec.impact(df)
    return res, dict(res.attrs)  # keep the replay details separately; cache copies drop attrs


def need_data():
    if not DEFAULT_CSV.exists():
        st.error("clearlink_service_tickets.csv must sit next to this script.")
        st.stop()
    return load_data(DEFAULT_CSV)


def candidate_chart(cand: pd.DataFrame, best_id: str) -> go.Figure:
    top = cand.head(8).iloc[::-1]  # best at the top
    colors = ["#15803D" if t == best_id else "#9CA3AF" for t in top["tech_id"]]
    labels = [f"{t} ({s})" for t, s in zip(top["tech_id"], top["seniority"])]
    fig = go.Figure(go.Bar(
        y=labels, x=top["predicted"], orientation="h", marker_color=colors,
        error_x=dict(type="data", symmetric=False,
                     array=top["high"] - top["predicted"],
                     arrayminus=top["predicted"] - top["low"], color="#6B7280"),
        hovertemplate="%{y}<br>Predicted %{x:.0f} min<extra></extra>",
    ))
    fig.add_vline(x=rec.TODAY_SLOT_MIN, line_dash="dash", line_color="#DC2626",
                  annotation_text="Today's 2-hour slot", annotation_position="top")
    fig.update_layout(height=60 + 38 * len(top), margin=dict(t=40, b=40, l=10),
                      xaxis_title="Predicted minutes (bar) and likely range (whisker)")
    return fig


def recommend_page():
    st.title("Who should take this job?")
    st.caption("Enter a job. The recommendation uses 18 months of ClearLink tickets: "
               "how long this job type takes, and how fast each tech in the zone does it.")
    df = need_data()
    stats = cached_stats(df)

    c1, c2, c3 = st.columns(3)
    job_type = c1.selectbox("Job type", sorted(df["job_type"].unique()))
    zone = c2.selectbox("Zone", sorted(df["zone"].unique()))
    last_day = df["date"].max().date()
    day = c3.date_input("Day", last_day, min_value=df["date"].min().date(),
                        max_value=last_day,
                        help="A day from the data, so we can see who already has jobs booked.")

    result = rec.recommend(stats, df, job_type, zone, day)
    cand = result["candidates"]
    best = cand.iloc[0].to_dict()

    if not best["has_room"]:
        st.warning("No tech in this zone has enough free time that day. Showing the best "
                   "option anyway; consider another day.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommended tech", best["tech_id"], best["seniority"], delta_color="off")
    m2.metric("Predicted duration", f"{best['predicted']:.0f} min",
              f"likely {best['low']:.0f}–{best['high']:.0f} min", delta_color="off")
    m3.metric("Book this slot", f"{best['slot']} min",
              f"{best['slot'] - rec.TODAY_SLOT_MIN:+d} min vs. today's 2 h",
              delta_color="off")
    m4.metric("Free time that day", f"{best['free_min'] / 60:.1f} h",
              f"{best['jobs_that_day']} other job(s)", delta_color="off")

    st.subheader("Why this tech")
    for reason in rec.explain(best, job_type, result["is_long"], cand):
        st.markdown(f"- {reason}")

    st.subheader("All techs in this zone")
    st.plotly_chart(candidate_chart(cand, best["tech_id"]), use_container_width=True)
    table = cand.assign(
        predicted=cand["predicted"].round(0).astype(int),
        likely_range=cand["low"].round(0).astype(int).astype(str) + "–" +
                     cand["high"].round(0).astype(int).astype(str),
        free_hours=(cand["free_min"] / 60).round(1),
    )[["tech_id", "seniority", "jobs_done", "predicted", "likely_range", "slot",
       "jobs_that_day", "free_hours", "has_room"]]
    st.dataframe(table.rename(columns={
        "tech_id": "Tech", "seniority": "Seniority", "jobs_done": "Past jobs of this type",
        "predicted": "Predicted min", "likely_range": "Likely range", "slot": "Slot to book",
        "jobs_that_day": "Jobs that day", "free_hours": "Free hours", "has_room": "Has room"}),
        use_container_width=True, hide_index=True)

    with st.expander("How the recommendation works"):
        st.markdown(
            "1. **Slot length:** for each job type and seniority level, book the time within "
            "which 80% of past jobs finished, rounded up to 15 minutes.\n"
            "2. **Predicted duration:** blend the tech's own average on this job type with the "
            "average for their seniority level. The more of these jobs they've done, the more "
            "their own history counts.\n"
            "3. **Choice:** long jobs go to the fastest tech with room that day. Short jobs, "
            "which anyone finishes inside 2 hours, go to the least senior tech with room, "
            "keeping seniors free for long jobs.\n"
            "4. **Room:** each tech has an 8-hour day; jobs already booked count at their "
            "right-sized slot length.")


def impact_page():
    st.title("What changes if ClearLink schedules this way?")
    df = need_data()
    res, a = cached_impact(df)
    st.caption(f"Rules learned from tickets before {a['test_from']:%b %Y}, then replayed on "
               f"the {a['test_jobs']:,} jobs from {a['test_from']:%b %Y} to "
               f"{a['test_to']:%b %Y}. The model never saw those jobs.")

    today, best = res.iloc[0], res.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("On-time today", f"{today['on_time']:.0%}")
    m2.metric("On-time with both changes", f"{best['on_time']:.0%}",
              f"{(best['on_time'] - today['on_time']) * 100:+.0f} pts")
    m3.metric("Extra booked time", f"{best['booked_hours'] / today['booked_hours'] - 1:+.0%}",
              "techs average 3.6 h of work a day today", delta_color="off")

    fig = go.Figure(go.Bar(
        x=res["on_time"], y=res["scenario"], orientation="h",
        marker_color=["#9CA3AF", "#1D4ED8", "#1D4ED8", "#15803D"],
        text=[f"{v:.0%}" for v in res["on_time"]], textposition="outside",
    ))
    fig.add_vline(x=0.80, line_dash="dash", line_color="#6B7280",
                  annotation_text="80% target", annotation_position="top")
    fig.update_layout(height=320, xaxis=dict(tickformat=".0%", range=[0, 1.05]),
                      yaxis=dict(autorange="reversed"), margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(res.assign(on_time=(res["on_time"] * 100).round(1),
                            booked_hours=res["booked_hours"].round(0).astype(int))
                 .rename(columns={"scenario": "Scenario", "on_time": "On-time %",
                                  "booked_hours": "Booked hours"}),
                 use_container_width=True, hide_index=True)

    st.subheader("Slot lengths ClearLink would book (minutes)")
    slots = cached_stats(df)["group"]["slot"].unstack()[["Junior", "Mid", "Senior"]]
    st.dataframe(slots, use_container_width=True)
    st.caption("Today every cell would be 120. Short jobs get shorter slots, which pays for "
               "part of the extra time long jobs need.")


st.set_page_config(page_title="ClearLink scheduling prototype", layout="wide")
st.navigation([
    st.Page(recommend_page, title="Recommend a tech", default=True),
    st.Page(impact_page, title="Impact"),
    st.Page(main, title="Data analysis"),
]).run()