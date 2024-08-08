#!/usr/bin/env python
# coding: utf-8

# ## Importing data
# 

# In[88]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[89]:


ESB_charger_data = pd.read_csv(r'C:\Users\92330\OneDrive\Desktop\Faraz\CV AND COVER LETTER\Data Analyst\ESB\Project\ESB charge point locations.csv')
EVregdata = pd.read_csv(r'C:\Users\92330\OneDrive\Desktop\Faraz\CV AND COVER LETTER\Data Analyst\ESB\Project\csv files\10 year parsed EV Registrations.csv')
ALL_chargers= pd.read_csv(r'C:\Users\92330\OneDrive\Desktop\Faraz\CV AND COVER LETTER\Data Analyst\ESB\Project\csv files\Total chargers by county(including ESB).csv')


# In[90]:


ESB_charger_data


# ## Data Analysis of EV Charge Points

# In[91]:


#getting ESB charger data

ESB_charger_data=ESB_charger_data.dropna(subset=['County'],axis=0)


# In[92]:


#creating a list of counties in Republic of Ireland so that we can exclude Nothern Ireland Chargers

list=['Sligo','Roscommon','Galway','Mayo','Leitrim','Wexford','Wicklow','Westmeath','Offaly','Meath','Laois','Louth','Longford','Kilkenny','Kildare','Dublin','Carlow','Limerick','Kerry','Tipperary','Waterford','Clare','Cork','Monaghan','Donegal','Cavan',]


# In[93]:


#creating dataframe

ChargersByCounty = pd.DataFrame(columns=['County', 'ESB_Charger_Count'])
ChargersByCounty


# In[94]:


#finding number of chargers per county from ESB data and inputting into Data Frame

i=0
for county in (list):
    df=ESB_charger_data[ESB_charger_data['County'].str.contains(list[i])]
    n = len(df)
    ChargersByCounty.loc[i] = [county,n]
    i=i+1
ChargersByCounty


# In[95]:


#merging column from data of ALL chargers county-wise into the existing to get 1 single data frame for analysis

ChargersByCounty['Total_Chargers']= ALL_chargers['Total_Chargers']
ChargersByCounty


# In[96]:


ChargersByCounty['NON_ESB_Chargers'] = ChargersByCounty['Total_Chargers'] - ChargersByCounty['ESB_Charger_Count'] 
ChargersByCounty


# In[97]:


#data cleaning for plot

charger_graph_data= ChargersByCounty.drop(columns=['Total_Chargers'])
charger_graph_data


# In[98]:


#plotting ESB chargers as 100% bar chart against non ESB chargers, county-wise

plt.figure(figsize=(20, 12))
charger_graph_data.plot.barh(x='County',stacked =True)
plt.tight_layout()
plt.show()


# ## Data Analysis of EV Registrations in Ireland

# In[99]:


#calculating total EV registrations in Ireland from county-wise data

AllIre = EVregdata.groupby(['Date'])['Total_Registrations'].sum().reset_index()
ir_graph_data = AllIre.sort_values(by= ['Date']) 
ir_graph_data = ir_graph_data[ir_graph_data.Total_Registrations !=0.0]
ir_graph_data
ir_graph_data['Date'] = pd.to_datetime(ir_graph_data.Date, format='%Y-%m-%d')
ir_graph_data['Year'] = ir_graph_data['Date'].dt.year
ir_graph_data


# In[100]:


#getting top 10 counties with most EV registrations since 2014

total_county_reg = EVregdata[EVregdata['Date'].str.contains('2024-07-01')].sort_values(by= ['Total_Registrations'], ascending = False)
top_ten_counties = total_county_reg.reset_index().drop(columns=['index','Registrations']).head(10)
top_ten_counties


# In[101]:


#creating a list of top 10 counties with most EV registrations for later use

top_ten_county_list = top_ten_counties['County'].tolist()
top_ten_county_list


# In[102]:


#plotting bar chart of  top 10 counties with most EV registration since 2014

plt.figure(figsize=(10, 6))
sns.barplot(x='County', y='Total_Registrations', data=top_ten_counties)
plt.title('Top 10 Counties by Total EV Registrations since 2014')
plt.tight_layout()


# In[103]:


#formatting date for graph

EVregdata['Date'] = pd.to_datetime(EVregdata.Date, format='%Y-%m-%d')
EVregdata


# In[104]:


#Plotting graph for top 10 county registrations (Seasonal) over time

plt.figure(figsize=(16, 9))
for county in (top_ten_county_list):
    county_data = EVregdata[EVregdata['County'].str.contains(county)]
    plt.plot(county_data['Date'], county_data['Registrations'], label=county)
    

plt.title('Trend of EV Registrations Over Time by County')
plt.xlabel('2014-2024')
plt.ylabel('Number of EV Registrations')
plt.legend(title='County', loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=5, fancybox=True, shadow=True) 
plt.grid(True)
plt.show()


# In[105]:


#Plotting graph for  top 10 county Total registrations (running sum) over time

plt.figure(figsize=(16, 9))
for county in (top_ten_county_list):
    county_data = EVregdata[EVregdata['County'].str.contains(county)]
    plt.plot(county_data['Date'], county_data['Total_Registrations'], label=county)
    

plt.title('Trend of Total EV Registrations Over Time by County')
plt.xlabel('2014-2024')
plt.ylabel('Total EV Registrations')
plt.legend(title='County', loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=5, fancybox=True, shadow=True) 
plt.grid(True)
plt.show()


# In[106]:


#plotting 10 year graph for trend for Ireland

plt.figure(figsize=(16, 9))
plt.plot(ir_graph_data['Date'], ir_graph_data['Total_Registrations'], label='Ireland')    
#plt.tick_params(labelbottom=False)
plt.title('Trend of Total EV Registrations Over Time in Ireland')
plt.xlabel('2014-2024')
plt.ylabel('Total EV Registrations')
plt.text(18800,113120, '113,121 Registrations since 2014', fontsize = 12)
plt.legend() 
plt.show()


# ## Train Test Split

# In[107]:


#getting specific country data:

def specific_county_data(name):
    df=EVregdata[EVregdata['County'].str.contains(name)].reset_index()
    df.drop(columns=['index'], inplace= True)
    return df 
specific_county_data('Dublin')


# In[108]:


#defining fuction for test split of Dublin Seasonal trend of EV Registrations 

def train_test_split_ts (df, train_size, test_size):
    
    """Function splits a given DataFrame into two sets based on the given 
    train and test sizes so that the data can be used for validation.
    -------------------------------
    Arguments:
    df: class: pandas.DataFrame
    The base dataframe that will be getting split.
    
    train_size: float
    The size of the desired training set (for example: 0.80)
    
    test_size: float
    The size of the desired training set (for example: 0.20)"""
    
    train_end_idx = int(round(len(df)*train_size,0))
    train_set = df.iloc[0:train_end_idx,:]["Registrations"]
    test_set = df.iloc[train_end_idx:,:]["Registrations"]
    return train_set, test_set


# In[109]:


#defining fuction for train split for Dublin EV Registrations (seasonal) since 2014

def plot_train_test_split(train_data, test_data, county):
    
    """Function plots the training and testing data for visual inspection.
    -------------------------------
    Arguments:
    train_data: pandas.Series
    The training set of data to be plotted.
    
    test_data: pandas.Series
    The test set of data to be plotted.
    
    county: str
    Name of the county that the training and testing data belongs to. This 
    string is used to set the title of the axes."""
    
    plt.figure(figsize=(10, 6))
   
    train_data.plot(label='Train Data')
    test_data.plot(label='Test Data')
    
    ax=plt.gca()
    ax.set_xlabel('Months')
    ax.set_ylabel('EV Registrations')
    ax.set_title(f'EV Registrations in {county} County')
    ax.legend();


# In[110]:


#splitting dataset into train and test sets for validation
train, test = train_test_split_ts(specific_county_data('Dublin'), 
                                            0.80, 0.20)
#plotting the split
plot_train_test_split(train, test, 'Dublin')


# In[111]:


#defining fuction for test split for Dublin trend of Total EV Registrations (running sum)

def train_test_split_tts (df, train_size, test_size):
    
    """Function splits a given DataFrame into two sets based on the given 
    train and test sizes so that the data can be used for validation.
    -------------------------------
    Arguments:
    df: class: pandas.DataFrame
    The base dataframe that will be getting split.
    
    train_size: float
    The size of the desired training set (for example: 0.80)
    
    test_size: float
    The size of the desired training set (for example: 0.20)"""
    
    train_end_idx = int(round(len(df)*train_size,0))
    train_set = df.iloc[0:train_end_idx,:]["Total_Registrations"]
    test_set = df.iloc[train_end_idx:,:]["Total_Registrations"]
    return train_set, test_set


# In[112]:


#defining fuction for train split for Dublin Total EV Registrations(running sum) since 2014

def plot_train_test_tsplit(train_data, test_data, county):
    
    """Function plots the training and testing data for visual inspection.
    -------------------------------
    Arguments:
    train_data: pandas.Series
    The training set of data to be plotted.
    
    test_data: pandas.Series
    The test set of data to be plotted.
    
    county: str
    Name of the county that the training and testing data belongs to. This 
    string is used to set the title of the axes."""
    
    plt.figure(figsize=(10, 6))
    
    train_data.plot(label='Train Data')
    test_data.plot(label='Test Data')
    
    ax=plt.gca()
    ax.set_xlabel('Months')
    ax.set_ylabel('Total EV Registrations')
    ax.set_title(f'Total EV Registrations in {county} County')
    ax.legend();


# In[113]:


#splitting dataset into train and test sets for validation
train, test = train_test_split_tts(specific_county_data('Dublin'), 
                                            0.80, 0.20)

#plotting the split
plot_train_test_tsplit(train, test, 'Dublin')


# ## MODEL

# In[114]:


#defining machine learning model

def get_forecast(model, train_data, test_data, plot=True):
    
    """Function gets forecasted values from a given model and plots them for 
    visual inspection. The length of the forecasts are dependent on the length 
    of the test data. The forecasted values are returned in a DataFrame format.
    -------------------------------
    Arguments:
    model:  SARIMAX or ARIMA model object
    Model that the forecast is to be received from. 
    
    train_data: pandas.Series
    The training set of data used in training the model.
    
    test_data: pandas.Series
    The testing set of data used for validating the model.
    
    plot: bool, default=True
    Option to plot the forecasted values along with observed values 
    (train_data and test_data).
    """
    
    #creating a df with the forecast information
    forecast_df = model.get_forecast(steps=len(test_data)).conf_int()
    forecast_df.columns = ['Lower Confidence Interval', 
                              'Upper Confidence Interval']
    forecast_df['Forecasts'] = model.get_forecast(steps=len(test_data))\
    .predicted_mean
    #plotting
    if plot==True:
        with plt.style.context('seaborn-v0_8-darkgrid'):
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.lineplot(data=train_data, color='black', ax=ax)
            sns.lineplot(data=forecast_df, x=forecast_df.index, 
                         y='Forecasts', color='blue', ax=ax, 
                         label='Forecasted Data', ls='--')
            sns.lineplot(data=test_data, color='purple', ax=ax, 
                         label='Actual Data', ls='-.')
            ax.fill_between(forecast_df.index, 
                            y1=forecast_df['Lower Confidence Interval'], 
                            y2=forecast_df['Upper Confidence Interval'],
                            color = 'green', alpha=0.3, 
                            label='Confidence Interval')
            ax.set_xlabel('Months')
            ax.legend(loc=2)
            plt.show();
    return forecast_df


# In[115]:


#defining machine learning model (cont.)

def get_prediction(model, df, test_data, county_name, plot=True):
    
    """Function gets predicted values from a given model and plots them for 
    visual inspection. The length of the predictions are dependent on the 
    length of the test data. The forecasted values are returned in a DataFrame 
    format.
    -------------------------------
    Arguments:
    model:  SARIMAX or ARIMA model object
    Model to be used for making predictions.
    
    df: pandas.DataFrame
    DataFrame that contains all observed data.
    
    test_data: pandas.Series
    The testing set of data used for validating the model (dictates the length
    of predictions).
    
    plot: bool, default=True
    Option to plot the predicted values along with observed values.
    """
    
    #creating a df with the prediction information
    prediction_df = model.get_forecast(steps=len(test_data)).conf_int()
    prediction_df.columns = ['Lower Confidence Interval', 
                              'Upper Confidence Interval']
    prediction_df['Predictions'] = model.get_forecast(steps=len(test_data))\
    .predicted_mean
    #plotting
    if plot==True:
        with plt.style.context('seaborn-v0_8-darkgrid'):
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.lineplot(data=df, ax=ax)
            sns.lineplot(data=prediction_df, x=prediction_df.index, 
                         y='Predictions', color='orange', ax=ax, 
                         label='Predicted Data', ls='--')
            ax.fill_between(prediction_df.index, 
                            y1=prediction_df['Lower Confidence Interval'], 
                            y2=prediction_df['Upper Confidence Interval'],
                            color = 'green', alpha=0.3, 
                            label='Confidence Interval')
            ax.set_xlabel('Months')
            ax.set_ylabel('Electric Vehicles on the Road')
            ax.set_title(f'Predicted Electric Vehicle Count for {county_name}')
            plt.show();
    return prediction_df


# In[116]:


#using Auto Arima for Feature Engineering and statistical modelling

get_ipython().system('pip install pmdarima')
import pmdarima as pm
auto_model = pm.auto_arima(train, start_p=0, d=1, start_q=0, max_p=4, 
                           max_d=3, max_q=4, start_P=0, start_Q=0, max_P=3, 
                           max_D=3, max_Q=3, m=12, seasonal = True)
auto_model.summary()


# In[117]:


#defining model summary

def evaluate_model(model):
    """Function returns the model summary and diagnostics information to aid 
    the evaluation of the given model's performance.
    -------------------------------
    Arguments:
    model: SARIMAX or ARIMA model object
    Model variable to evaluate (Time series models for both pmdarima and 
    statsmodels are supported. 
    """
    
    display(model.summary())
    model.plot_diagnostics()
    plt.tight_layout();


# In[118]:


#since order of (1,1,1) is the best choice as per SARIMAX Statistical model, we use it for the evaluation

from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train, order=(1,1,1),  enforce_invertibility=False,enforce_stationarity=False).fit()
evaluate_model(model)


# In[119]:


#forecasting future EV registrations

df_dublin_forecast = get_forecast(model, train, test, plot=True)


# In[120]:


#using SARIMAX for capturing the forecast

model = SARIMAX(specific_county_data('Dublin').drop(columns=['County', 'Year' ,'Month', 'Date', 'Registrations']), order=(1,1,1), enforce_invertibility=False,enforce_stationarity=False).fit()
evaluate_model(model)
#specific_county_data('Dublin')


# In[121]:


#creating a df of predictions and plotting

df_preds = get_prediction(model, specific_county_data('Dublin').drop(columns=['County', 'Year' ,'Month', 'Date', 'Registrations']) , test, 'Dublin', plot=True)


# In[122]:


#saving predictions df to dict for later use

df_dublin_pred= specific_county_data('Dublin')
df_preds.insert(0, 'County',["Dublin"]*len(df_preds))
df_preds


# In[123]:


df_preds.drop([156,157], axis=0, inplace=True)


# In[124]:


Date=['Jan25','Feb25','Mar25','Apr25','May25','Jun25','July25','Aug25','Sep25','Oct25','Nov25','Dec25','Jan26','Feb26','Mar26','Apr26','May26','Jun26','July26','Aug26','Sep26','Oct26','Nov26','Dec26']
df_preds.insert(1, 'Dates',Date)


# In[125]:


df_dublin= df_preds.reset_index().drop(columns=['index'])


# In[126]:


df_dublin

