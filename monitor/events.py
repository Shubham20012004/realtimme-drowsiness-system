"""One-to-one onset matching in seconds within a single independently annotated session."""
from statistics import median


def match_events(truth, alerts, observed_seconds):
    if observed_seconds<=0:
        raise ValueError("Positive observed duration required")
    ordered=sorted(truth)
    if any(end<=start for start,end in ordered) or any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):
        raise ValueError("Expected disjoint event intervals")
    remaining=sorted(alerts)
    latencies=[]
    for start,end in ordered:
        i=next((i for i,t in enumerate(remaining) if start<=t<end),None)
        if i is not None:
            latencies.append(remaining.pop(i)-start)
    return dict(events=len(truth),detected=len(latencies),missed=len(truth)-len(latencies),
                event_recall=len(latencies)/len(truth) if truth else None,
                false_alarms_per_hour=len(remaining)*3600/observed_seconds,
                median_latency_s=median(latencies) if latencies else None)
