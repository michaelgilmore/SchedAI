"""
Scheduling logic for the ClearLink prototype. Plain pandas, no machine learning,
so every number can be checked by hand.

1. Slot length: for each job type and seniority level, book the time within
   which 80% of past jobs finished, rounded up to 15 minutes.
2. Predicted duration for a specific tech: blend the tech's own average on this
   job type with the average for their seniority level. The more of these jobs
   the tech has done, the more weight their own history gets.
3. Who gets the job:
   - Long jobs (ones that often overrun 2 hours): the fastest tech in the zone
     who still has room in their day.
   - Short jobs (anyone finishes inside 2 hours): the least senior tech with
     room, so seniors stay free for the long jobs.
"""

import math

import numpy as np
import pandas as pd

TODAY_SLOT_MIN = 120          # what ClearLink books for every job today
TARGET_PCTL = 0.80            # slot = time within which 80% of similar jobs finished
ROUND_TO_MIN = 15
SHRINK_K = 10                 # a tech's own history counts fully after ~10 jobs
WORKDAY_MIN = 8 * 60          # bookable minutes per tech per day
SENIORITY_ORDER = {"Junior": 0, "Mid": 1, "Senior": 2}


def round_up(minutes: float) -> int:
    return int(math.ceil(minutes / ROUND_TO_MIN) * ROUND_TO_MIN)


def build_stats(hist: pd.DataFrame):
    """Group stats per (job type, seniority) and per (tech, job type)."""
    group = (
        hist.groupby(["job_type", "tech_seniority"])["actual_duration_min"]
        .agg(n="count", mean="mean",
             p20=lambda s: s.quantile(0.20), p80=lambda s: s.quantile(TARGET_PCTL),
             fits_today=lambda s: (s <= TODAY_SLOT_MIN).mean())
    )
    group["slot"] = group["p80"].apply(round_up)
    tech = (
        hist.groupby(["tech_id", "job_type"])["actual_duration_min"]
        .agg(n="count", mean="mean")
    )
    techs = hist.groupby("tech_id")[["tech_seniority", "zone"]].first()
    # A job type is "long" if even the slowest seniority level usually overruns 2 hours.
    long_jobs = set(group[group["fits_today"] < 0.9].index.get_level_values("job_type"))
    return {"group": group, "tech": tech, "techs": techs, "long_jobs": long_jobs}


def predict(stats, tech_id: str, job_type: str) -> dict:
    """Predicted duration and slot for one tech on one job type."""
    sen = stats["techs"].loc[tech_id, "tech_seniority"]
    g = stats["group"].loc[(job_type, sen)]
    if (tech_id, job_type) in stats["tech"].index:
        t = stats["tech"].loc[(tech_id, job_type)]
        n, own = int(t["n"]), float(t["mean"])
    else:
        n, own = 0, float(g["mean"])
    w = n / (n + SHRINK_K)
    pred = w * own + (1 - w) * g["mean"]
    shift = pred - g["mean"]  # move the seniority-level range by this tech's difference
    return {
        "tech_id": tech_id, "seniority": sen, "jobs_done": n, "own_avg": own,
        "level_avg": float(g["mean"]), "predicted": float(pred),
        "low": float(g["p20"] + shift), "high": float(g["p80"] + shift),
        "slot": round_up(g["p80"] + shift),
        "fits_today_pct": float(g["fits_today"]),
    }


def recommend(stats, hist: pd.DataFrame, job_type: str, zone: str, day) -> dict:
    """Rank the zone's techs for a new job on a given day."""
    zone_techs = stats["techs"][stats["techs"]["zone"] == zone].index
    day_jobs = hist[hist["date"].dt.date == day]
    rows = []
    for tid in zone_techs:
        p = predict(stats, tid, job_type)
        booked = day_jobs[day_jobs["tech_id"] == tid]
        # Time already committed that day, using right-sized slots for those jobs.
        used = sum(predict(stats, tid, jt)["slot"] for jt in booked["job_type"])
        p.update(jobs_that_day=len(booked), booked_min=used,
                 free_min=WORKDAY_MIN - used, has_room=WORKDAY_MIN - used >= p["slot"])
        rows.append(p)
    cand = pd.DataFrame(rows)
    is_long = job_type in stats["long_jobs"]
    if is_long:
        cand = cand.sort_values(["has_room", "predicted"], ascending=[False, True])
    else:
        cand["rank_sen"] = cand["seniority"].map(SENIORITY_ORDER)
        cand = cand.sort_values(["has_room", "rank_sen", "predicted"],
                                ascending=[False, True, True]).drop(columns="rank_sen")
    return {"candidates": cand.reset_index(drop=True), "is_long": is_long}


def explain(best: dict, job_type: str, is_long: bool, cand: pd.DataFrame) -> list:
    """Plain-English reasons for the recommendation."""
    reasons = []
    if is_long:
        slowest = cand["predicted"].max()
        reasons.append(f"{job_type} often runs past 2 hours, so it goes to the fastest "
                       f"tech in the zone with room today.")
        reasons.append(f"{best['tech_id']} is predicted to take {best['predicted']:.0f} min; "
                       f"the slowest option in this zone would take about {slowest:.0f} min.")
    else:
        reasons.append(f"{job_type} almost always finishes inside 2 hours, so a less senior "
                       f"tech is chosen to keep senior techs free for long jobs.")
    if best["jobs_done"] > 0:
        reasons.append(f"{best['tech_id']} has done {best['jobs_done']} of these jobs, "
                       f"averaging {best['own_avg']:.0f} min (the {best['seniority'].lower()} "
                       f"average is {best['level_avg']:.0f} min).")
    else:
        reasons.append(f"{best['tech_id']} has no history on this job type, so the "
                       f"{best['seniority'].lower()} average ({best['level_avg']:.0f} min) is used.")
    if best["slot"] > TODAY_SLOT_MIN:
        today = f"Today's 2-hour slot fits only {best['fits_today_pct']:.0%} of them."
    else:
        today = (f"Today's 2-hour slot would leave about "
                 f"{TODAY_SLOT_MIN - best['predicted']:.0f} min unused.")
    reasons.append(f"Book {best['slot']} min: 80% of similar jobs finished within that "
                   f"time. {today}")
    reasons.append(f"{best['tech_id']} has {best['jobs_that_day']} other job(s) that day and "
                   f"{best['free_min'] / 60:.1f} h free before this one.")
    return reasons


def impact(df: pd.DataFrame, split="2026-03-01") -> pd.DataFrame:
    """Replay the last months under each rule, learning only from the earlier months."""
    train, test = df[df["date"] < split], df[df["date"] >= split]
    key = ["job_type", "tech_seniority"]
    slot = (train.groupby(key)["actual_duration_min"].quantile(TARGET_PCTL)).apply(round_up)
    mean = train.groupby(key)["actual_duration_min"].mean()
    durations = {k: np.sort(g.to_numpy()) for k, g in train.groupby(key)["actual_duration_min"]}

    def p_fit(jt, sen, s):  # share of past jobs of this kind that finished within s minutes
        d = durations[(jt, sen)]
        return np.searchsorted(d, s, side="right") / len(d)

    pairs = list(zip(test["job_type"], test["tech_seniority"]))
    # Skill-aware re-match inside each day and zone: hardest jobs to most senior techs.
    hard = mean.groupby("job_type").mean().sort_values(ascending=False).index
    hard_rank = {j: i for i, j in enumerate(hard)}
    sen_rank = {"Senior": 0, "Mid": 1, "Junior": 2}
    rematched = []
    for _, g in test.groupby(["date", "zone"]):
        jobs = sorted(g["job_type"], key=hard_rank.get)
        people = sorted(g["tech_seniority"], key=sen_rank.get)
        rematched += list(zip(jobs, people))

    actual = test["actual_duration_min"].to_numpy()
    rows = [
        ("Today: 2-hour slot for every job", float((actual <= TODAY_SLOT_MIN).mean()),
         len(test) * TODAY_SLOT_MIN / 60),
        ("A. Slot sized to job type and seniority",
         float((actual <= np.array([slot[p] for p in pairs])).mean()),
         sum(slot[p] for p in pairs) / 60),
        ("B. Skill-aware assignment, 2-hour slot",
         float(np.mean([p_fit(j, s, TODAY_SLOT_MIN) for j, s in rematched])),
         len(test) * TODAY_SLOT_MIN / 60),
        ("A + B together", float(np.mean([p_fit(j, s, slot[(j, s)]) for j, s in rematched])),
         sum(slot[p] for p in rematched) / 60),
    ]
    out = pd.DataFrame(rows, columns=["scenario", "on_time", "booked_hours"])
    out.attrs.update(test_jobs=len(test), test_from=test["date"].min(), test_to=test["date"].max())
    return out
