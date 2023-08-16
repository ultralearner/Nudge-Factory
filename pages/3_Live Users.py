import streamlit as st
import pandas as pd
import time
from collections import Counter, defaultdict
from datetime import datetime

import plotly.express as px

from nudge_engine import process_trigger

# Display the banner
st.image("pics/nudgex-banner.png")


@st.cache_data
def load_data():
    return pd.read_csv('data/users_synthetic_dataset.csv')


df = load_data()
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.sort_values(by='Timestamp', inplace=True)

# Initialize counters, accumulators, and sets
category_counter = Counter()
trigger_counter = Counter()
users_set = set()
accuracy_accumulator_onboarding = defaultdict(list)
accuracy_accumulator_engagement = defaultdict(list)
accuracy_accumulator_feedback = defaultdict(list)

# Create a new DataFrame to hold the Nudge Effectiveness scores and current time
nudge_effectiveness_data = pd.DataFrame(
    columns=['Time', 'Nudge_Effectiveness'])

st.subheader('User Activity and Nudge Simulation')
progress_bar = st.progress(0)

col_start, col_stop = st.columns(2)
start = col_start.button('START')
stop = col_stop.button('STOP')

# Print a divider line just above the container for the messages
st.markdown('---')

# Create six columns at the top for the metrics
metric1, metric2, metric3, metric4, metric5, metric6 = st.columns(6)
metric1 = metric1.empty()
metric2 = metric2.empty()
metric3 = metric3.empty()
metric4 = metric4.empty()
metric5 = metric5.empty()
metric6 = metric6.empty()

# print the graph title as NUDGE EFFECTIVENESS TREND in Bold and centered
st.markdown('<p style="text-align: center;"><strong>NUDGE EFFECTIVENESS TREND</strong></p>',
            unsafe_allow_html=True)


# Row 1 with placeholders
graphs1 = st.columns(1)
nudge_effectiveness_trend_plot_placeholder = graphs1[0].empty()

# Print a divider line just above the container for the messages
st.markdown('---')

current_user_column = st.columns(1)
current_user_placeholder = current_user_column[0].empty()

# Create three columns below the graph
nudge_column, explanation_column, rating_column = st.columns(3)
nudge_placeholder = nudge_column.empty()
explanation_placeholder = explanation_column.empty()
rating_placeholder = rating_column.empty()

# Print a divider line just above the container for the messages
st.markdown('---')

st.markdown('USER ACTIVITY LOG')
user_activity_container = st.container()

start_time = None
if start:
    start_time = datetime.now()

    for index, row in df.iterrows():
        if stop:
            st.write('Stopped.')
            break

        progress_bar.progress(index / len(df))
        category_counter[row["Category"]] += 1
        trigger_counter[row["Trigger"]] += 1
        users_set.add(row["User_ID"])

        nudge, explanation, accuracy, f1, predicted_rating = process_trigger(
            row["User_ID"], row["Asset_ID"], row["Trigger"])

        if accuracy is not None:
            if row["Category"] == "Onboarding":
                accuracy_accumulator_onboarding[row["Category"]].append(
                    accuracy)
            elif row["Category"] == "Engagement":
                accuracy_accumulator_engagement[row["Category"]].append(
                    accuracy)
            elif row["Category"] == "Feedback":
                accuracy_accumulator_feedback[row["Category"]].append(accuracy)

       # Get the current time
        current_time = datetime.now()
        nudge_effectiveness_data = nudge_effectiveness_data.append({
            'Time': current_time,
            'Nudge_Effectiveness': row['Nudge_Effectiveness']
        }, ignore_index=True)

        fig = px.line(nudge_effectiveness_data,
                      x='Time', y='Nudge_Effectiveness')
        nudge_effectiveness_trend_plot_placeholder.plotly_chart(fig)

        # Update the current user with the latest activity in the column; format User ID  and Asset ID in bold
        current_user_placeholder.markdown(
            f"**{current_time.strftime('%H:%M:%S')}** - User <span style='color: blue; font-weight: bold;'>{row['User_ID']}</span> interacted with Asset <span style='color: green; font-weight: bold;'>{row['Asset_ID']}</span>. Category: **{row['Category']}**. Trigger: **{row['Trigger']}**.", unsafe_allow_html=True)
        with user_activity_container:
            st.markdown(
                f'**{current_time.strftime("%H:%M:%S")}** - User {row["User_ID"]} interacted with Asset {row["Asset_ID"]}. Category: **{row["Category"]}**. Trigger: **{row["Trigger"]}**.')

        # Update the metrics in the columns
        metric1.metric('# Users', len(users_set))
        metric2.metric('# Nudges issued', sum(trigger_counter.values()))

        if start_time:
            elapsed_time = datetime.now() - start_time
            minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
            metric3.metric(
                'Time Elapsed', f'{int(minutes):02}:{int(seconds):02}')

        metric4.metric('Onboarding accuracy', round(sum(accuracy_accumulator_onboarding.get(
            "Onboarding", [0])) / (len(accuracy_accumulator_onboarding.get("Onboarding", [])) or 1), 3))
        metric5.metric('Engagement accuracy', round(sum(accuracy_accumulator_engagement.get(
            "Engagement", [0])) / (len(accuracy_accumulator_engagement.get("Engagement", [])) or 1), 3))
        metric6.metric('Feedback accuracy', round(sum(accuracy_accumulator_feedback.get(
            "Feedback", [0])) / (len(accuracy_accumulator_feedback.get("Feedback", [])) or 1), 3))

        # Inside  loop, after calling process_trigger:
        nudge_placeholder.markdown(
            f"<span style='color: purple; font-weight: bold; font-size: 14px;'>Nudge:</span> <span style='font-size: 12px;'>{nudge}</span>",
            unsafe_allow_html=True)
        explanation_placeholder.markdown(
            f"<div style='color: green; font-weight: bold; font-size: 14px;'>Explanation:</div><div style='font-size: 12px;'>{explanation}</div>",
            unsafe_allow_html=True
        )
        rating_placeholder.markdown(
            f"<span style='color: blue; font-weight: bold; font-size: 14px;'>Predicted Rating:</span> <span style='font-size: 12px;'>{predicted_rating:.2f}</span>",
            unsafe_allow_html=True)

        # Create some delay for simulating real-time data
        time.sleep(0.1)
