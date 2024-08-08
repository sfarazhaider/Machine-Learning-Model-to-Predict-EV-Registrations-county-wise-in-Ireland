# Machine-Learning-Model-to-Predict-EV-Registrations-county-wise-in-Ireland

![image](https://github.com/user-attachments/assets/db3291c0-6894-43f2-82e2-8ce45101c008)

# Summary
This project aims to understand the rising demand of EV Chargers in Ireland as EV Registrations for in county are rising rapidly (most notably in Dublin). By Training a machine learning statistical model, a 10 year data for EV Registrations for each county is obtained, trained and tested. Then, with the help of Auto Arima and SARIMAX, future EV registrations are predicted for the next 2 years for each county. As per the European Union suggestion, the ratio of EV to EV Charger is ideally 10:1. With the help of this model, we can pre plan and proactively manage the infrastructure accordingly for each county.

# Data
There are 3 data sets used for this project
1. ESB charge point locations.csv    source: ESB Website
2. 10 year parsed EV Registrations   source: SIMI | The Society of the Irish Motor Industry
3. Total chargers by county(including ESB)   source:  https://www.electromaps.com/en/charging-stations/ireland

# Results
![ESB chargers vs Total Chargers](https://github.com/user-attachments/assets/95b7a9a7-a896-4006-933a-a730126931b8)

![Top 10 Counties by Total EV Registrations](https://github.com/user-attachments/assets/76222f71-6a63-4e7f-b26a-1e291d509776)

![Trend of EV Registrations overtime by County](https://github.com/user-attachments/assets/aba76b55-47cc-4ad5-8be4-9d7ccd2aa1fc)

![Trend of Total EV Registrations overtime in Ireland](https://github.com/user-attachments/assets/0595b62e-15c0-488e-8f1c-b8ebf52870b0)






![EV Registrations in Dublin](https://github.com/user-attachments/assets/1921a3bb-48aa-4cb9-8b2f-f658db455dd7)





# Limitations Recommendations
- Since API data sets were not readily available, parsing for datasets had to be done by scraping through MS Power Automate, in the future, an API based data set should be used for obtaining data sets of EV registrations with more granuality. For example, EV registrations weekly transaction data,  make and model of each of them, tpye of EV, new or used and also charging activity. This would tranform the model to depp learning for more accurate forecasts.
