# open-webui-cost-tracker

This open-webui function is designed to manage and calculate the costs associated with user interactions with models in a Open WebUI appliance.
The number of (input and output) tokens, price and other metrics are shown in the message status area of Open Webui interface. This status appear right above the message content (see Fig. below)

![image](https://github.com/user-attachments/assets/1d7a975b-84b2-4af1-93b3-2bfb3363f575)


![image](https://github.com/user-attachments/assets/ad373135-9ead-465f-adb1-d10f6262a705)

## Install

Remember to enable the function globally (or by model): 

![image](https://github.com/user-attachments/assets/07cb5d0e-f6eb-4e5b-98dd-6d29510af972)

## Streamlit App

This repo also provides a streamlit app to read and process the generated `costs.json` file containing model usage data for every user. 
The `costs-<year>.json` file is located in the open-webui `data` directory. 

A screenshot of the simple streamlit app is shown below. 

[Streamlit app for costs processing](https://open-webui-cost-tracker.streamlit.app)

![image](https://github.com/user-attachments/assets/9529ae29-9dd6-4295-8417-9371430f8a88)
