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
            cost = record['total_cost']
            total_tokens = record['input_tokens'] + record['output_tokens']
            processed_data.append({
                'month': month,
                'model': model,
                'total_cost': cost,
                'user': user,
                'total_tokens': total_tokens
            })
    return pd.DataFrame(processed_data)

def plot_data(data, month):
    """Plot the data for a specific month."""
    month_data = data[data['month'] == month]

    # Model Usage Bar Plot (Total Tokens)
    month_data_models_tokens = month_data.groupby('model')['total_tokens'].sum().reset_index()
    month_data_models_tokens = month_data_models_tokens.sort_values(by='total_tokens', ascending=False)
    fig_models_tokens = px.bar(month_data_models_tokens, x='model', y='total_tokens', title=f'Total Tokens Used for {month} by Model')
    st.plotly_chart(fig_models_tokens, use_container_width=True)

    # User Cost Bar Plot (Total Cost)
    month_data_users = month_data.groupby('user')['total_cost'].sum().reset_index()
    month_data_users = month_data_users.sort_values(by='total_cost', ascending=False)
    month_data_users['total'] = month_data_users['total_cost'].sum()
    month_data_users = pd.concat([month_data_users, pd.DataFrame({'user': ['Total'], 'total_cost': [month_data_users['total'].iloc[0]]})])
    fig_users = px.bar(month_data_users, x='user', y='total_cost', title=f'Cost for {month} by User')
    st.plotly_chart(fig_users, use_container_width=True)

    # Model Cost Bar Plot (Total Cost)
    month_data_models_cost = month_data.groupby('model')['total_cost'].sum().reset_index()
    month_data_models_cost = month_data_models_cost.sort_values(by='total_cost', ascending=False)
    fig_models_cost = px.bar(month_data_models_cost, x='model', y='total_cost', title=f'Total Cost for {month} by Model')
    st.plotly_chart(fig_models_cost, use_container_width=True)

def main():
    st.title("Open Webui")
    st.subheader("Cost Tracker App", divider=False)
    st.page_link("https://github.com/bgeneto/open-webui-cost-tracker/", label="App Home", icon="🏠")
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
