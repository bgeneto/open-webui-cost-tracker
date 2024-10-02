import json
import plotly.express as px
import pandas as pd
import streamlit as st
from datetime import datetime

def load_data(file):
    """Load data from a JSON file."""
    try:
        data = json.load(file)
        return data
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid JSON file.")
        return None

def process_data(data):
    """Process the data by extracting the month, model, cost, and user."""
    processed_data = []
    for user, records in data.items():
        for record in records:
            timestamp = datetime.strptime(record['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
            month = timestamp.strftime('%Y-%m')
            model = record['model']
            cost = record['cost']
            processed_data.append({
                'month': month,
                'model': model,
                'cost': cost,
                'user': user
            })
    return pd.DataFrame(processed_data)

def plot_data(data, month):
    """Plot the data for a specific month."""
    month_data = data[data['month'] == month]
    month_data_models = month_data.groupby('model')['cost'].sum().reset_index()
    month_data_models['total'] = month_data_models['cost'].sum()
    month_data_models = pd.concat([month_data_models, pd.DataFrame({'model': ['Total'], 'cost': [month_data_models['total'].iloc[0]]})])
    fig_models = px.bar(month_data_models, x='model', y='cost', title=f'Cost for {month} by Model')
    st.plotly_chart(fig_models, use_container_width=True)

    month_data_users = month_data.groupby('user')['cost'].sum().reset_index()
    month_data_users['total'] = month_data_users['cost'].sum()
    month_data_users = pd.concat([month_data_users, pd.DataFrame({'user': ['Total'], 'cost': [month_data_users['total'].iloc[0]]})])
    fig_users = px.bar(month_data_users, x='user', y='cost', title=f'Cost for {month} by User')
    st.plotly_chart(fig_users, use_container_width=True)

def main():
    st.title("Open Webui")
    st.subtitle("User Cost Tracker App")
    file = st.file_uploader("Upload a JSON file", type=["json"])
    if file is not None:
        data = load_data(file)
        if data is not None:
            processed_data = process_data(data)
            months = processed_data['month'].unique()
            month = st.selectbox("Select a month", months)
            if st.button("Plot Data"):
                plot_data(processed_data, month)

if __name__ == "__main__":
    main()
