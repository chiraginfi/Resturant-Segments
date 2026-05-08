import pandas as pd
import ast
import re




def extract_timings(x):
    try:
        lst = ast.literal_eval(x)
        timings = []

        for item in lst:
            # case 1: proper JSON
            if isinstance(item, dict):
                timings.append(item.get('timing'))

            # case 2: string like {weekday=Sunday, timing=10AM–11:30PM}
            else:
                match = re.search(r'timing[=:]([^,}]+)', item)
                if match:
                    timings.append(match.group(1))

        return timings
    except:
        return []
def convert(x, fallback_period=None):
    x = x.strip().upper()

    match = re.match(r'(\d{1,2})(?::(\d{2}))?(AM|PM)?', x)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    period = match.group(3)

    # 🔥 if AM/PM missing → infer from other side
    if period is None:
        period = fallback_period

    if period is None:
        return None

    if period == 'PM' and hour != 12:
        hour += 12
    if period == 'AM' and hour == 12:
        hour = 0

    return hour + minute/60

def classify_operational(timings):
    if not timings:
        return None

    # 🔥 HANDLE 24 HOURS FIRST
    for t in timings:
        if t and '24' in str(t).lower():
            return "All-Day Operational"

    def to_24h(t):
        try:
            t = t.replace('–', '-')
            start, end = t.split('-')

            # detect AM/PM from end
            end_match = re.search(r'(AM|PM)', end.upper())
            fallback_period = end_match.group(1) if end_match else None

            def convert(x, fallback_period=None):
                x = x.strip().upper()

                match = re.match(r'(\d{1,2})(?::(\d{2}))?(AM|PM)?', x)
                if not match:
                    return None

                hour = int(match.group(1))
                minute = int(match.group(2)) if match.group(2) else 0
                period = match.group(3) or fallback_period

                if period is None:
                    return None

                if period == 'PM' and hour != 12:
                    hour += 12
                if period == 'AM' and hour == 12:
                    hour = 0

                return hour + minute/60

            s = convert(start, fallback_period)
            e = convert(end, fallback_period)

            # 🔥 handle overnight
            if s is not None and e is not None and e < s:
                e += 24

            return s, e
        except:
            return None, None

    starts, ends = [], []

    for t in timings:
        s, e = to_24h(t)
        if s is not None and e is not None:
            starts.append(s)
            ends.append(e)

    if not starts:
        return None

    avg_start = sum(starts)/len(starts)
    avg_end = sum(ends)/len(ends)

    # classification
    if avg_start <= 9 and avg_end <= 22:
        return "All-Day Operational"
    elif avg_start >= 15 and avg_end <= 24:
        return "Evening Operational"
    elif avg_start >= 20 or avg_end >= 24:
        return "Late-Night Operational"
    else:
        return "Mixed"


df = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step9_output.csv")

# apply
df['timings_list'] = df['open_close_hours'].apply(extract_timings)
df['operational_status'] = df['timings_list'].apply(classify_operational)

print(df[['timings_list', 'operational_status']])

# optional: drop helper
# df.drop(columns=['timings_list'], inplace=True)

df.to_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step10_output.csv", index=False)