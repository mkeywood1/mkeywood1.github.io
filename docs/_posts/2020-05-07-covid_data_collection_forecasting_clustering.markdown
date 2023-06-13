---
layout: post
title:  "Project: COVID-19 Data Collection, Forecasting & Clustering"
date:   2020-05-07 19:26:28 +0100
categories: data forecasting clustering
---

**NOTE: After this project was done, COVID-19 progressed at a devastating rate, effecting the lives of millions in tragic and deeply upsetting ways. This project was not meant to trivialise COVID-19 but in fact to try and help people understand the onslaught of data they were starting to see.
I put it together at the start of the first UK lockdown, but the UK data was not as available as US data and so for illustrative purposes this focused on the US.**

Utilising Real World Data, can we look at forecasting cases, and also clustering States into similar characteristics of their COVID-19 profile.

Having trawled a few sites, and looked at what data is out there, I came up with the following set of sources that I think would be useful to start to collate data across things like Covid stats, details of what public service / places are open / closed, other prevelant clinical conditions per state etc:

| Source                       | Source                                                                                                                                                                                           | Description                           | Level  | Update Frequency |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- | ------ | ---------------- |
| Google Mobility data         | gstatic.com/covid19/mobility/Global_Mobility_Report.csv                                                       | Google Mobility data                  | County | Adhoc            |
| John  Hopkins - Confirmed US | raw.githubusercontent.com/CSSEGISandData/COVID-19/master/ csse_covid_19_data/csse_covid_19_time_series/ time_series_covid19_confirmed_US.csv                                               | Confirmed Cases - Time Series         | County | Daily            |
| John  Hopkins - Deaths US    | raw.githubusercontent.com/CSSEGISandData/COVID-19/master/ csse_covid_19_data/csse_covid_19_time_series/ time_series_covid19_deaths_US.csv                                                  | Confirmed Deaths - Time Series        | County | Daily            |
| Ancilliary data              | Misc_ML.csv                                                                                                                                                                                      | Used to get State code from Full Name | State  | N/A              |
| CovidTracking.com            | covidtracking.com/api/v1/states/daily.csv                                                                                           | Daily Covid information               | State  | Daily            |
| Boston University            | docs.google.com/spreadsheets/d/1zu9qEWI8PsOI_i8nI_ S29HDGHlIp2lfVMsGxpQ5tvAQ/edit#gid=0 | Covid collation re State Policy       | State  | Adhoc            |
| Chronic Conditions           | Chronic_Conditions_Prevalence_2017.xlsx                                                                                                                                                          | Chronic Condition Prevalence          | County | Adhoc            |
| 2016 Vote data               | raw.githubusercontent.com/tonmcg/US_County_Level_Election_ Results_08-16/master/US_County_Level_Presidential_Results_12-16.csv                                                            | 2016 electorial vote data             | County | N/A              |


This started as a personal project but we went on to take what I had started and used the data to do some more advanced forecasting and analytics for commercial uses.

So what did I start with in my personal project?

## First obviously the usual imports, and then define some functions for the main Feature Engineering and Forecasting

{% highlight python %}
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')
{% endhighlight %}

### The `get_all_data` function is the guts of the Feature Engineering
Here we pull in, cleanse and merge to a State level for:
- Google Mobility data
- Confirmed Covid case data
- Covid death data
- Hospitalisation data
- Employment / Homlessness rates
- Chronic Conditions for over 65's
- 2016 Vote Data
- State Wide StayAtHome / Closure rules

All normalised out to per 100k of the State population

{% highlight python %}
def get_all_data(path):
    """Compile mobility, case, and policy data for COVID-19 modeling.

    Inputs:
    path           - File path

    Outputs:
    df_all_data    - County-level time series dataset
    df_state_data  - State-level summaries
    """
    
    ##############################
    # Get the Google Mobility data
    ##############################
    df_google_mobility = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv', low_memory=False)
    df_google_mobility = df_google_mobility.loc[df_google_mobility['country_region'] == 'United States']
    df_google_mobility = df_google_mobility.drop(df_google_mobility[df_google_mobility['sub_region_1'].isna()].index)
    df_google_mobility = df_google_mobility.drop(df_google_mobility[df_google_mobility['sub_region_2'].notna()].index)
    df_google_mobility.rename({'retail_and_recreation_percent_change_from_baseline' : 'retail',
                           'grocery_and_pharmacy_percent_change_from_baseline' : 'grocery.and.pharmacy',
                           'parks_percent_change_from_baseline' : 'parks',
                           'transit_stations_percent_change_from_baseline' : 'transit.stations',
                           'workplaces_percent_change_from_baseline' : 'workplaces',
                           'residential_percent_change_from_baseline': 'residential'}, axis=1, inplace=True)
    df_google_mobility.to_csv(path+'Data/Google_Mobility_Report.csv', index=False)
    
    #################
    # Get df_all_data
    #################
    df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
    #.to_csv(path+'Data/df_confirmed.csv', index=False)
   
    df_confirmed.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_'], axis=1, inplace=True)
    
    df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
    #df_deaths.to_csv(path+'Data/df_deaths.csv', index=False)
    
    metadata = df_deaths[['Combined_Key', 'Lat', 'Long_', 'Province_State', 'Population']]
    
    df_deaths.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_', 'Population'], axis=1, inplace=True)
    
    # Sum up df_confirmed per County
    df_tmp = df_confirmed.groupby('Combined_Key')[df_confirmed.columns[1:]].sum()
    df_tmp = pd.DataFrame(df_tmp.T.stack()).reset_index()
    df_tmp.columns = ['Date', 'Combined_Key', 'Confirmed']
    df_tmp = df_tmp[df_tmp['Confirmed'] > 1]
    df_tmp['Confirmed_size'] = df_tmp['Confirmed'].pow(0.5)
    
    # Sum up df_deaths per County
    df_tmp1 = df_deaths.groupby('Combined_Key')[df_deaths.columns[1:]].sum()
    # Transpose and Stack, etc
    df_tmp1 = pd.DataFrame(df_tmp1.T.stack()).reset_index()
    df_tmp1.columns = ['Date', 'Combined_Key', 'Deaths']
    df_tmp1 = df_tmp1[df_tmp1['Deaths'] > 1]
    df_tmp1['Deaths_size'] = df_tmp1['Deaths'].pow(0.5)
    
    df_all_data = df_tmp.merge(df_tmp1, on=['Combined_Key', 'Date'], how='left').fillna(0)
    df_all_data['Date'] = pd.to_datetime(df_all_data['Date'], format="%m/%d/%y")
    df_all_data['Date'] = df_all_data['Date'].dt.date
    df_all_data['Date'] = df_all_data['Date'].astype(str)
    df_all_data = df_all_data.sort_values('Date', ascending=True)
    df_all_data.Date = pd.Categorical(df_all_data.Date)
    df_all_data["Day"] = df_all_data["Date"].cat.codes

    df_all_data = df_all_data.merge(metadata, on='Combined_Key')
    
    df_all_data = df_all_data.merge(df_google_mobility[['retail', 'grocery.and.pharmacy', 'parks', 'transit.stations',
                                             'workplaces', 'residential', 'sub_region_1', 'date']], 
                                            left_on=['Province_State', 'Date'], 
                                            right_on=['sub_region_1', 'date'],
                                            how='left').drop(['sub_region_1', 'date'], axis=1).fillna(0)

    df_AW_data = pd.read_csv(path+'Data/Misc_ML.csv')
    df_AW_data = df_AW_data[['state', 'NAME', 'test_per_100k']]
        
    df_all_data = df_all_data.merge(df_AW_data, left_on='Province_State', right_on='NAME').drop(['NAME'], axis=1)
    df_all_data[['Confirmed_size', 'Deaths_size']] = df_all_data[['Confirmed_size', 'Deaths_size']].astype(float)
    
    
    ##########################
    # Add Hospitalisation Data
    ##########################
    df_tmp_hospitalised = pd.read_csv('https://covidtracking.com/api/v1/states/daily.csv', parse_dates=['date'], usecols=['date', 'state', 'hospitalizedCurrently', 'inIcuCurrently', 'onVentilatorCurrently'])
    df_tmp_hospitalised.dropna(subset=['hospitalizedCurrently', 'inIcuCurrently'], how='all', inplace=True)
    df_tmp_hospitalised.rename({'date' : 'Date'}, axis=1, inplace=True)
    df_tmp_hospitalised = df_tmp_hospitalised.fillna(0)
    df_tmp_hospitalised.loc[df_tmp_hospitalised.hospitalizedCurrently == 0 , 'hospitalizedCurrently'] = df_tmp_hospitalised.inIcuCurrently
    df_tmp_hospitalised['Date'] = pd.to_datetime(df_tmp_hospitalised['Date'], format="%Y-%m-%d")
    df_tmp_hospitalised['Date'] = df_tmp_hospitalised['Date'].dt.date
    df_tmp_hospitalised['Date'] = df_tmp_hospitalised['Date'].astype(str)
    df_all_data = df_all_data.merge(df_tmp_hospitalised, on=['state', 'Date'], how='left').fillna(0)

    
    ######################
    # Add Demographic Data
    ######################
    df_tmp_demog = pd.read_csv(path+'Data/COVID-19 US state policy database.csv', usecols=['State',
                                                                                          'Number Homeless (2019)',
                                                                                          'Percent Unemployed (2018)',
                                                                                          'Percent living under the federal poverty line (2018)',
                                                                                          'Percent at risk for serious illness due to COVID',
                                                                                          'All-cause deaths 2016']).dropna()
    df_tmp_demog.rename({'State' : 'Province_State',
                        'Number Homeless (2019)' : 'HomelessNumber',
                        'Percent Unemployed (2018)' : 'EmployedPerc',
                        'Percent living under the federal poverty line (2018)' : 'PovertyPerc',
                        'Percent at risk for serious illness due to COVID' : 'SeriousIllnessRiskPerc',
                        'All-cause deaths 2016' : 'AllCauseDeaths'}, axis=1, inplace=True)
    df_all_data = df_all_data.merge(df_tmp_demog, on=['Province_State'], how='left')
    
    
    ######################################
    # Add Chronic Conditions for over 65's
    ######################################
    df_tmp_chronic = pd.read_excel(path+'Data/Chronic_Conditions_Prevalence_2017.xlsx', sheet_name='Beneficiaries 65 Years and Over')
    df_tmp_chronic = df_tmp_chronic.drop(df_tmp_chronic.loc[df_tmp_chronic['County'] == '  '].index)
    df_tmp_chronic = df_tmp_chronic.replace("* ", 0)
    df_tmp_chronic['State'] = df_tmp_chronic['State'].apply( lambda x : x.split()[0] )
    df_tmp_chronic['County'] = df_tmp_chronic['County'].apply( lambda x : x.split()[0] )
    df_all_data['County'] = df_all_data['Combined_Key'].apply( lambda x : x.split(",")[0] )
    df_all_data = df_all_data.merge(df_tmp_chronic, left_on=['Province_State','County'], right_on=['State','County'], how='left').drop('State', axis=1)
    
    
    ####################
    # Add 2016 Vote Data
    ####################
    df_tmp_votes = pd.read_csv('https://raw.githubusercontent.com/tonmcg/US_County_Level_Election_Results_08-16/master/US_County_Level_Presidential_Results_12-16.csv',
                            usecols=['votes_dem_2016', 'votes_gop_2016', 'state_abbr', 'county_name'])
    df_tmp_votes['county_name'] = df_tmp_votes['county_name'].apply( lambda x : x.split()[0] )
    df_tmp_votes.drop_duplicates(inplace=True)
    df_all_data = df_all_data.merge(df_tmp_votes, left_on=['state','County'], right_on=['state_abbr','county_name'], how='left').drop(['state_abbr','county_name'], axis=1)


    ###################
    # Get df_state_data
    ###################
    df_tmp = df_all_data.loc[df_all_data['Date'] == max(df_all_data['Date']),['Population', 'Confirmed', 'Deaths']].groupby(df_all_data['state']).sum().astype(int)
    
    df_tmp1 = df_all_data.loc[df_all_data['Date'] == max(df_all_data['Date']),['Date', 'Province_State', 'state']].drop_duplicates()
    
    df_state_data = df_tmp.merge(df_tmp1,on='state')
    df_state_data['Deaths_size'] = df_state_data['Deaths'].pow(0.5)
    df_state_data['Confirmed_size'] = df_state_data['Confirmed'].pow(0.5)
    df_state_data.to_csv(path+'Data/df_state_data.csv', index=False)


    ############################################################################################################################
    # Collate State Wide StayAtHome, BusinessesClosed, RestaurantsClosed, GymsClosed, CinemasClosed, SuspendedElectiveProcedures
    ############################################################################################################################
    df_tmp_restrictions = pd.read_csv(path+'Data/COVID-19 US state policy database.csv', usecols=['State',
                                            'Stay at home/ shelter in place', 'End/relax stay at home/shelter in place',
                                            'Closed non-essential businesses', 'Reopen businesses', 
                                            'Closed restaurants except take out', 'Reopen restaurants',
                                            'Closed gyms', 'Repened gyms',
                                            'Closed movie theaters', 'Reopened movie theaters',
                                            'Suspended elective medical/dental procedures', 'Resumed elective medical procedures']).dropna()
    df_tmp_restrictions.rename({'State' : 'Province_State',
                        'Stay at home/ shelter in place' : 'StayAtHome_Start',
                        'End/relax stay at home/shelter in place' : 'StayAtHome_End',
                        'Closed non-essential businesses' : 'BusinessesClosed_Start',
                        'Reopen businesses' : 'BusinessesClosed_End',
                        'Closed restaurants except take out' : 'RestaurantsClosed_Start',
                        'Reopen restaurants' : 'RestaurantsClosed_End',
                        'Closed gyms' : 'GymsClosed_Start',
                        'Repened gyms' : 'GymsClosed_End',
                        'Closed movie theaters' : 'CinemasClosed_Start',
                        'Reopened movie theaters' : 'CinemasClosed_End',
                        'Suspended elective medical/dental procedures' : 'SuspendedElectiveProcedures_Start',
                        'Resumed elective medical procedures' : 'SuspendedElectiveProcedures_End'}, axis=1, inplace=True)
    
    df_tmp_TS = pd.DataFrame(columns =  ['Date', 'Province_State', 'StayAtHome', 'BusinessesClosed', 'RestaurantsClosed', 'GymsClosed', 'CinemasClosed', 'SuspendedElectiveProcedures'])
    row_index = 0
    
    for state in df_state_data['Province_State']:
        for restriction in ['StayAtHome', 'BusinessesClosed', 'RestaurantsClosed', 'GymsClosed', 'CinemasClosed', 'SuspendedElectiveProcedures']:
            start = df_tmp_restrictions.loc[df_tmp_restrictions['Province_State'] == state, restriction+'_Start'].values[0]
            if not start == "0":
                start = pd.to_datetime(start)
                end = df_tmp_restrictions.loc[df_tmp_restrictions['Province_State'] == state, restriction+'_End'].values[0]
                if end == "0":
                    end = pd.to_datetime('today')
                else:
                    end = pd.to_datetime(end)
                    
                for dt in pd.date_range(start=start, end=end, freq='d'):
                    existingIndexes = df_tmp_TS.loc[(df_tmp_TS['Date']==dt) & (df_tmp_TS['Province_State']==state)].index
                    if len(existingIndexes) == 0:
                        df_tmp_TS.loc[row_index, 'Date'] = dt
                        df_tmp_TS.loc[row_index, 'Province_State'] = state
                        df_tmp_TS.loc[row_index, restriction] = 1
                        row_index+=1
                    else:
                        df_tmp_TS.loc[existingIndexes[0], restriction] = 1
              
    df_tmp_TS['Date'] = df_tmp_TS['Date'].astype('datetime64[ns]')
    df_tmp_TS['Date'] = df_tmp_TS['Date'].dt.date
    df_tmp_TS['Date'] = df_tmp_TS['Date'].astype(str)
    #df_tmp_TS = df_tmp_TS.fillna(0)
    df_all_data = df_all_data.merge(df_tmp_TS, on=['Date', 'Province_State'], how='left').fillna(0)

    df_all_data.to_csv(path+'Data/df_all_data.csv', index=False)
      
    return df_all_data, df_state_data
{% endhighlight %}

### The 'get_forecast' function executes a simple ARIMA forecast for a desired data point

{% highlight python %}
def get_forecast(path, day0_feature, forecast_days):
    
    """Generate forecasts for COVID-19 case and death time series.

    Inputs:
    path             - File path
    day0_feature     - Column name for start of time series
    forecast_days    - Number of days to forecast

    Outputs:
    df_forecast_data - Forecasted time series
    """
    
    df_all_states = pd.read_csv(path+'Data/df_all_states.csv')
    df_all_states_TS = pd.read_csv(path+'Data/df_all_states_TS.csv')
    
    df_forecast_data = pd.DataFrame()

    for state in df_all_states['Province_State']:
        #print(state)

        plot_data = df_all_states_TS.loc[df_all_states_TS['Province_State'] == state, ['Date', day0_feature]]#.reset_index().drop('index', axis=1)
        plot_data.loc[plot_data[day0_feature] == 0, day0_feature] = plot_data[day0_feature].mean()

        today = max(plot_data.Date)
        index = pd.date_range(start=today, periods=forecast_days, freq="D")
        idxdates = index[1:]

        # row_index = max(plot_data.index)+1

        X = plot_data[day0_feature].values
        history = [x for x in X]
        forecast = []

        # Perform the forecast
        for t in range(forecast_days-1):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            history.append(yhat)
            forecast.append(yhat)


        # Calculate the details that will be appended and used for any plotting
        plot_data['state'] = df_all_states_TS.loc[df_all_states_TS['Province_State'] == state, ['state']].values[0][0]
        plot_data['Province_State'] = state
        plot_data['size'] = plot_data[day0_feature]
        plot_data.loc[plot_data['size'] < 0, ['size']] = 0

        # Tabularise the predictions, and associated information to allow merging of all the data at the end
        preds = pd.DataFrame(data=forecast, columns=["forecast"])
        preds['Date'] = idxdates[:len(preds)]
        preds['Date'] = preds['Date'].dt.date
        preds['Date'] = preds['Date'].astype(str)
        preds['state'] = df_all_states_TS.loc[df_all_states_TS['Province_State'] == state, ['state']].values[0][0]
        preds['Province_State'] = state
        preds['size'] = preds['forecast']

        plot_data = plot_data.append(preds)

        df_forecast_data = df_forecast_data.append(plot_data)

    df_forecast_data.to_csv(path+'Data/df_forecast_data_'+day0_feature+'.csv', index=False)    

    return df_forecast_data
{% endhighlight %}

## OK let's get started

### First, generate or load the data

{% highlight python %}
get_fresh_data = False            # Toggle this to pull fresh data, else use an already prepared version
regenerate_forecast_data = False  # Regenerate the forecast data (will be overwritten to True if getFreshData == True)

path = ''

if get_fresh_data:
    df_all_data, df_state_data = get_all_data(path)
else:
    df_all_data = pd.read_csv(path+'Data/df_all_data.csv')
    df_state_data = pd.read_csv(path+'Data/df_state_data.csv')
{% endhighlight %}

### Let's take a look at df_all_data

{% highlight python %}
fig, ax = plt.subplots(figsize=(8, 6))

df_tmp = df_all_data.groupby(['Date', 'state'])[['Confirmed', 'Deaths']].sum()
df_tmp = df_tmp.reset_index()
df_tmp = df_tmp.sort_values(['state', 'Date'])
df_tmp['Date'] = pd.to_datetime(df_tmp['Date'])

# Get state names and iterate 
states = df_tmp['state'].unique()
for state in states:
    ax.plot(df_tmp[df_tmp['state']==state]['Date'], df_tmp[df_tmp['state']==state]['Confirmed'], label='Confirmed')
    
    # Get dates on which state changes
    dates = df_tmp.loc[df_tmp['state'] == state, 'Date'].unique()
    
    # Annotate last date of state 
    plt.annotate(state, (dates[-1], df_tmp.loc[(df_tmp['Date'] == dates[-1]) & 
                                         (df_tmp['state'] == state), 
                                         'Confirmed'].values[0]), xytext=(5, 0), textcoords='offset points') 
plt.xlabel('Date')
plt.ylabel('Number of cases')
# Rotate the x-axis labels by 45 degrees
ax.set_xticks(ax.get_xticks())
ax.tick_params(axis='x', rotation=45)
plt.show()
{% endhighlight %}
<img src="https://mkeywood1.github.io/2020-05-07-covid_data_collection_forecasting_clustering/1.png">

{% highlight python %}

{% endhighlight %}



{% highlight python %}

{% endhighlight %}



{% highlight python %}

{% endhighlight %}



Thanks for reading :-)
