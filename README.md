# open-webui-cost-tracker

This open-webui function is designed to manage and calculate the costs associated with user interactions with models in a Open WebUI appliance.
The number of (input and output) tokens, price and other metrics are shown in the message status area of Open Webui interface. This status appear right above the message content (see Fig. below)

![image](https://github.com/user-attachments/assets/ad373135-9ead-465f-adb1-d10f6262a705)

This repo also provides a streamlit app to read and process the generated `costs.json` file containing model usage data for every user. 
The `costs-<year>.json` file is located in the open-webui `data` directory. 
A screenshot of the simple streamlit app is shown below. 


[streamlit app](https://open-webui-costs.streamlit.app)
