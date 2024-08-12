# Machine-Learning-Model-to-Predict-EV-Registrations-county-wise-in-Ireland

![image](https://github.com/user-attachments/assets/db3291c0-6894-43f2-82e2-8ce45101c008)

image source:https://www.geekwire.com/2023/ev-charging-startup-electric-era-lands-11-5m-to-deploy-ai-and-battery-supported-stations/

# Summary
This project aims to understand the rising demand of EV Chargers in Ireland as EV Registrations for each county are rising rapidly (most notably in Dublin). The data set for EV registrations is obtained from SIMI (Society of Irish Motor Indusry) and it also includes plug-in hybrids which are also relevant in the EV-charging landscape. By training a machine learning statistical model on a month-wise 10 year dataset for EV Registrations for each county, Train-Test split is performed. Then, with the help of Auto Arima and SARIMAX, I find the best model fit while catering to the seasonality of my dataset. The dependent variable y is total running sum of EV registrations since 2014. Finally, for 104 number of observations, with the right SARIMAX model fit of parameters (1,1,1) we get log likelihood of -951.867. Based on this fit, future EV registrations are predicted for the next 2 years for each county. With the help of this model, we can pre plan and proactively manage the infrastructure accordingly for each county.

# Data
There are 3 data sets used for this project
1. ESB charge point locations.csv    source: ESB Website
2. 10 year parsed EV Registrations   source: SIMI | The Society of the Irish Motor Industry
3. Total chargers by county(including ESB)   source:  https://www.electromaps.com/en/charging-stations/ireland

# Results
![ESB chargers vs Total Chargers](https://github.com/user-attachments/assets/95b7a9a7-a896-4006-933a-a730126931b8)

![Top 10 Counties by Total EV Registrations](https://github.com/user-attachments/assets/76222f71-6a63-4e7f-b26a-1e291d509776)

![Trend of EV Registrations overtime by County](https://github.com/user-attachments/assets/aba76b55-47cc-4ad5-8be4-9d7ccd2aa1fc)

![Trend of Total EV Registrations overtime by County](https://github.com/user-attachments/assets/c43701cf-2005-4a5e-998c-1e8f8d1d6251)

![Trend of Total EV Registrations overtime in Ireland](https://github.com/user-attachments/assets/0595b62e-15c0-488e-8f1c-b8ebf52870b0)

![EV Registrations in Dublin](https://github.com/user-attachments/assets/f884f26e-05a0-43d1-8e56-09eb2581b69b)

![Total EV Registrations in Dublin since 2014](https://github.com/user-attachments/assets/f86eabae-bc65-40d5-bd38-c3e8975bb767)

![Forecasted Total EV Registrations](https://github.com/user-attachments/assets/613b26d5-863e-4143-8656-5fe79b206a7c)






# Limitations Recommendations
- Since API data sets were not readily available, parsing for datasets had to be done by scraping through MS Power Automate, in the future, an API based data set should be used for obtaining data sets of EV registrations with more granuality. For example, weekly EV registrations transaction data to increase number of observations,  make and model, type of EV, new or used, and also charging activity. This would transform the model to deep learning for more accurate forecasts.
