---
layout: post
title:  "Project: COVID-19 Data Collection, Forecasting & Clustering"
date:   2020-05-07 19:26:28 +0100
categories: data forecasting clustering
---

**NOTE: After this project was done, COVID-19 progressed at a devastating rate, effecting the lives of millions in tragic and deeply upsetting ways. This project was not meant to trivialise COVID-19 but in fact to try and help people understand the onslaught of data they were starting to see.
I put it together at the start of the first UK lockdown, but the UK data was not as available as US data and so for illustrative purposes this focused on the US.**

This started as a personal project but we went on to take what I had started and used the data to do some more advanced forecasting and analytics for commercial uses.

So what did I start with in my personal project?

## Utilising Real World Data, can we look at forecasting cases, and also clustering States into similar characteristics of their COVID-19 profile?

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

All normalised out to per 100k of the State population.

There is a lot of good data manipulation and Feature Engineering here, but all fairly straight forward.

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
	
    df_all_data = df_all_data.merge(df_tmp_TS, on=['Date', 'Province_State'], how='left').fillna(0)

    df_all_data.to_csv(path+'Data/df_all_data.csv', index=False)
      
    return df_all_data, df_state_data
{% endhighlight %}

### The `get_forecast` function executes a simple ARIMA forecast for a desired data point

ARIMA (AutoRegressive Integrated Moving Average) is a time series forecasting method that models the relationship between variables over time.

This function is again fairly straight forward. We forecast for the number of days requested, and return all the data in a tidy DataFrame.

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

        plot_data = df_all_states_TS.loc[df_all_states_TS['Province_State'] == state, ['Date', day0_feature]]#.reset_index().drop('index', axis=1)
        plot_data.loc[plot_data[day0_feature] == 0, day0_feature] = plot_data[day0_feature].mean()

        today = max(plot_data.Date)
        index = pd.date_range(start=today, periods=forecast_days, freq="D")
        idxdates = index[1:]

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
    
    # Get dates
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

We can see that New York is way out ahead in terms of COVID-19 cases, second being New Jersey, which makes sense from a geographic / commuting sense.

Then California looks to be taking a swift upturn, and certainly needs to be kept an eye on.

### Then let's look at some of the conditions for infected people

{% highlight python %}
fig, ax = plt.subplots(figsize=(10, 6))

outcome_cols = ['notHospitalized', 'hospitalizedCurrently', 'inIcuCurrently', 'onVentilatorCurrently', 'Deaths']

# First get the latest totals per state for all except 'Confirmed'
df_outcomes = df_all_data.groupby(['state'])[outcome_cols[1:]].last()
df_outcomes = df_outcomes.reset_index()

# Confirmed is a daily count so to compare apples and apples we need to make a cumulative count column
df_tmp = df_all_data.groupby(['state'])['Confirmed'].sum().reset_index()

df_outcomes = df_outcomes.merge(df_tmp, on="state")
df_outcomes.drop("state", axis=1, inplace=True)

df_outcomes['notHospitalized'] = df_outcomes['Confirmed'] - (df_outcomes['hospitalizedCurrently'] + df_outcomes['Deaths'])
df_outcomes.drop("Confirmed", axis=1, inplace=True)

plt.pie(df_outcomes.sum(), labels=df_outcomes.sum().index, autopct='%1.1f%%')
plt.title('Case Outcomes')
plt.show()
{% endhighlight %}

<img src="https://mkeywood1.github.io/2020-05-07-covid_data_collection_forecasting_clustering/2.png">

### We can see the Not Hospitalized counts are dwarfing the rest, so let's excude those

{% highlight python %}
fig, ax = plt.subplots(figsize=(10, 6))

plt.pie(df_outcomes[outcome_cols[1:]].sum(), labels=df_outcomes[outcome_cols[1:]].sum().index, autopct='%1.1f%%')
plt.title('Case Outcomes')
plt.show()
{% endhighlight %}

<img src="https://mkeywood1.github.io/2020-05-07-covid_data_collection_forecasting_clustering/3.png">

We can see from these two charts that the number of infected people who are hospitalised is <1%, and the resultant deaths are <1% of those - still way too many :-(

### Let's take a look at df_state_data

{% highlight python %}
fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(df_state_data['Confirmed'], df_state_data['Deaths'], 
            s=df_state_data['Population']/100000, alpha=0.5)

df_tmp = df_state_data.query('Confirmed > 50000 | Deaths > 4500')

for x, y, z in zip(df_tmp['Confirmed'], df_tmp['Deaths'], df_tmp['state']):
    ax.annotate(z, (x, y), fontsize=12, color='red', weight='bold')
    
plt.xlabel('Total Cases')
plt.ylabel('Total Deaths')
plt.show()
{% endhighlight %}

<img src="https://mkeywood1.github.io/2020-05-07-covid_data_collection_forecasting_clustering/4.png">

We can see the same picture with regards to those states heavily effected. The size of the bubble corelates to the population, so we can see with the size of California coupled with visable increase seen in the last graph again means this is something to keep a close eye on.

## OK let's go on to collate the forecast data, and see if we can forecast the next 10 days

{% highlight python %}
if regenerate_forecast_data or get_fresh_data:
    df_all_states, df_all_states_TS = get_forecast_data(path, day0_feature, day0_minimum, checkpoint_interval)
else:
    df_all_states = pd.read_csv(path+'Data/df_all_states.csv')
    df_all_states_TS = pd.read_csv(path+'Data/df_all_states_TS.csv')
{% endhighlight %}

### Firstly the number of positive cases

{% highlight python %}
def show_forecast(state):
    fig, ax = plt.subplots(figsize=(12, 6))

    df_tmp = df_forecast_data[df_forecast_data["state"]==state]

    ax.plot(df_tmp["Date"], df_tmp[day0_feature])
    ax.plot(df_tmp["Date"], df_tmp["forecast"])
    # ax.set_xticks(ax.get_xticks())
    # ax.tick_params(axis='x', rotation=90)

    # set the number of tick labels on the x-axis
    num_ticks = 7
    xtick_locs = ax.get_xticks()
    xtick_labels = [loc for loc in xtick_locs[::int(len(xtick_locs)/num_ticks)]]
    xtick_labels = [df_tmp.iloc[loc]['Date'] for loc in xtick_locs[::int(len(xtick_locs)/num_ticks)]]
    ax.set_xticks(xtick_locs[::int(len(xtick_locs)/num_ticks)])
    ax.set_xticklabels(xtick_labels)

    plt.show()
	
day0_minimum = 10
forecast_days = 10

day0_feature = 'positive'
df_forecast_data = get_forecast(path, day0_feature, forecast_days)
# df_forecast_data

# Let's look at the forecast for North Carolina
show_forecast("NC")
{% endhighlight %}

<img src="https://mkeywood1.github.io/2020-05-07-covid_data_collection_forecasting_clustering/5.png">

### And what about forecasting the increase rate

{% highlight python %}
day0_feature = 'positiveIncrease'
df_forecast_data = get_forecast(path, day0_feature, forecast_days)
# df_forecast_data

# Let's look at the forecast for North Carolina
show_forecast("NC")
{% endhighlight %}

<img src="https://mkeywood1.github.io/2020-05-07-covid_data_collection_forecasting_clustering/6.png">

Unsuprisingly, the forecast of both values shows an upwards trend at the time of generation :-(

## And finally, can we cluster the states into similarity in terms of it's Covid characteristics etc

First a helper function to allow us to pull out the fields pertinent to the clustering.

{% highlight python %}
def cluster_columns_only(path, checkpoint_interval):
    """return just the fields we are going to use for K-Means clustering.

    Inputs:
    path                - File path
    checkpoint_interval - Interval between checkpoints

    Outputs:
    df_all_states       - State-level summaries with cluster assignments
    X                   - Data of just the relevant columns
    """

    df_all_states = pd.read_csv(path+'Data/df_all_states.csv')
    try:
        df_all_states.drop('Cluster', axis=1, inplace=True)
    except:
        pass

    toExclude = ['Province_State', 'state', 'MaxDays', 'Population', 'Confirmed', 'Deaths']

    X = df_all_states.copy()
    X.drop(list(toExclude), axis=1, inplace=True)

    toExclude = []
    for chk in range(1,10):
        for col in X.columns:
            if col.split('_')[0] == 'Day'+str(chk*checkpointInterval):
                toExclude.append(col)
    X.drop(toExclude, axis=1, inplace=True)
    X = X.astype(str).replace(',','', regex=True)
    X = X.astype(float)
    
    return df_all_states, X
{% endhighlight %}

And it's appropriate pipeline for the preprocessing.
- Scaling is important for K-Means because K-Means is a distance-based algorithm that clusters data points based on their Euclidean distance from a centroid. If the features in the dataset are not scaled, some of them may be given higher weights than others, which can result in clustering biases towards features with larger magnitudes. This can lead to poor cluster assignments and reduced accuracy
- Principal Component Analysis (PCA) is a dimensionality reduction technique that is used to reduce the number of features in a dataset, while retaining as much of the original variance and information in the dataset as possible. Therefore PCA can be useful for preprocessing data before clustering

{% highlight python %}
preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)
{% endhighlight %}

Then get the training data columns, and pre-process them.

{% highlight python %}
checkpointInterval = 7

df_all_states, X = cluster_columns_only(path, checkpointInterval)

preprocessed_X = preprocessor.fit_transform(X)
# preprocessed_X
{% endhighlight %}

## But how can we determine the right number of clusters?

There are two good methods we can use to try and determine the appropriate value for k. The elbow method and silhouette method.
- The elbow method involves plotting the relationship between the number of clusters (k) and the sum of squared distances between data points and their assigned cluster center. The ideal k value is the point where the decrease in sum of squared distances starts to level off (i.e., the elbow point)
- The silhouette method involves calculating the average silhouette score for each number of clusters k. The silhouette score measures how similar a data point is to its assigned cluster compared to other clusters. It ranges from -1 to 1, where values closer to 1 indicate that a data point is well matched to its cluster

### Elbow method to select k

{% highlight python %}
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 1972,
}

# A list holds the sum of squared errors (SSE) values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(preprocessed_X)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
{% endhighlight %}

<img src="https://mkeywood1.github.io/2020-05-07-covid_data_collection_forecasting_clustering/7.png">

### Silhouette method to select k

{% highlight python %}
# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice we start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(preprocessed_X)
    score = silhouette_score(preprocessed_X, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
{% endhighlight %}

<img src="https://mkeywood1.github.io/2020-05-07-covid_data_collection_forecasting_clustering/8.png">

Both methods indicate **3** is a good cluster size, so let's build it with that.

{% highlight python %}
n_clusters = 3

clusterer = Pipeline(
   [
       (
           "kmeans",
           KMeans(
               n_clusters=n_clusters,
               init="k-means++",
               n_init=50,
               max_iter=500,
               random_state=42,
           ),
       ),
   ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)

pipe.fit(X)
{% endhighlight %}

Get the predicted labels, and the associated silhoutte score.

{% highlight python %}
predicted_labels = pipe["clusterer"]["kmeans"].labels_
predicted_labels
{% endhighlight %}

`array([1, 2, 2, 1, 0, 2, 0, 0, 0, 1, 0, 2, 0, 1, 2, 1, 1, 1, 2, 0, 0, 0,
       2, 1, 1, 2, 2, 0, 2, 0, 2, 0, 1, 2, 1, 1, 2, 0, 0, 1, 2, 1, 1, 2,
       2, 0, 2, 1, 2, 2])`

{% highlight python %}
silhouette_score(preprocessed_X, predicted_labels)
{% endhighlight %}

`0.4398675185111555`

So silhouette ranges from -1 to 1, so this is not bad - pretty confident :-)

### Finally let's look at these as clusters

{% highlight python %}
pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(X),
    columns=["component_1", "component_2"],
)

# Add in the cluster
pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_

# Add in the State code
pcadf['state'] = df_all_states['state'].to_list()
pcadf.head()
{% endhighlight %}

<img src="https://mkeywood1.github.io/2020-05-07-covid_data_collection_forecasting_clustering/9.png">

{% highlight python %}
plt.figure(figsize=(8, 8))

scat = sns.scatterplot(
    "component_1",
    "component_2",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    palette="Set1",
)

# loop through the data and add annotations for each point
for i in range(len(pcadf)):
    label = pcadf.iloc[i]['state']  # get the label from the 'state' column in the data
    x = pcadf.iloc[i]['component_1']  # get the x-coordinate
    y = pcadf.iloc[i]['component_2']  # get the y-coordinate
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')


scat.set_title("Clustering results for States with Covid Characteristic Similarities")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()
{% endhighlight %}

<img src="https://mkeywood1.github.io/2020-05-07-covid_data_collection_forecasting_clustering/10.png">

This looks pretty good to me. Cluster 0 has NY, NJ, CA etc, which makes sense, and I think there are some good insights here as to how things could progress.

Thanks for reading :-)
