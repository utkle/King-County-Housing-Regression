# Standard Packages
import pandas as pd
import numpy as np
import datetime
import json

# # Viz Packages
# import seaborn as sns
# import matplotlib.pyplot as plt

# Scipy Stats
import scipy.stats as stats 

# # Statsmodel Api
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

# # SKLearn Modules
# from sklearn.linear_model import LinearRegression
# from sklearn.feature_selection import RFE
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split
# import sklearn.metrics as metrics

# Folium and Streamlit
import streamlit as st
from streamlit_folium import folium_static
import folium


# Pull csv into DataFrame, dataset provided
df = pd.read_csv('data/kc_house_data.csv')

# Create column for zipcode
df['zipcode'] = 1
for i in range(len(df['address'])):
    df['zipcode'][i] = df['address'][i][-20:-15]
df['zipcode'] = df['zipcode'].astype(int)

# Pull list of zipcodes associatied with King County, dataset from King County
z = pd.read_csv('data/Zipcodes_for_King_County_and_Surrounding_Area_(Shorelines)___zipcode_shore_area.csv')
z = z[z['COUNTY_NAME'] == 'King County']

# Filter house data DataFrame for zipcodes within County
in_king_county_mask = df['zipcode'].isin(z['ZIPCODE'])
df_king = df[in_king_county_mask]
df_king = df_king.drop_duplicates()

# Classify houses based on 'condition' and 'grade'
# Create ordinal numerics for both 'condition' and 'grade'
dict = {'Poor':1, 'Fair':2, 'Average':3, 'Good':4, 'Very Good':5}
df_king=df_king.replace({"condition": dict})

df_king['gradeno'] = df_king['grade'].map(lambda x: x[0:2])
df_king['gradeno'] = df_king['gradeno'].astype('int')
df_king = df_king.drop(columns = 'grade')

# Create class column
def get_class(row):
    """Classify properties into 3 groups. 0 includes all 'Poor' and 'Fair' condition rows, and well as 'Average' condition where grade is 7 or less.
    Class 2 includes all 'Very good' condition rows, as well as 'Good' condition where grade is 9 or more.
    Class 1 includes all else."""
    if row['condition'] <= 2:
        return 0
    elif row['condition'] == 3 and row['gradeno'] <= 7:
        return 0
    elif row['condition'] == 5:
        return 2
    elif row['condition'] == 4 and row['gradeno'] >= 9:
        return 2
    else:
        return 1

df_king['class'] = df_king.apply(get_class, axis=1)

# Find difference between class 2 and class 0 median property values per zip, neglecting zip codes with null values
df_median_price = pd.pivot_table(df_king, values='price', index='zipcode', columns='class', aggfunc='median')
df_median_price = df_median_price.dropna()
df_median_price = df_median_price.reset_index()
df_median_price['diff'] = df_median_price[2] - df_median_price[0]
df_median_price = df_median_price.rename(columns={'zipcode': 'ZIPCODE'})

# Map difference per zip, zipcode border data from Kings County
geojson_file = 'data/Zipcodes_for_King_County_and_Surrounding_Area_(Shorelines)___zipcode_shore_area.geojson'
zipcodes = df_median_price['ZIPCODE'].tolist()
diffs = df_median_price['diff'].tolist()

# Load the geojson data
with open(geojson_file) as f:
    geojson_data = json.load(f)

# Add the 'diff' data to the geojson properties
for feature in geojson_data['features']:
    zipcode = int(feature['properties']['ZIP'])
    if zipcode in zipcodes:
        index = zipcodes.index(zipcode)
        feature['properties']['diff'] = diffs[index]

# Create the map
map = folium.Map(location=[df_king['lat'].mean(), df_king['long'].mean()], zoom_start=8)

# Be able to create transparent layers
def style_function(feature):
    return {
        'fillOpacity': 0,
    }

# Add the geojson data to the map
folium.GeoJson(geojson_data, name = 'ZIPs', style_function=style_function).add_to(map)

# Create the choropleth layer
folium.Choropleth(
    geo_data=geojson_data,
    name='diff',
    data=df_median_price,
    columns=('ZIPCODE', 'diff'),
    key_on='feature.properties.ZIP',
    fill_color='YlOrRd',
    fill_opacity=0.8,
    nan_fill_opacity=0,
    line_opacity=1,
    highlight=True,
    legend_name='Difference in median price high/low quality'
).add_to(map)

# Add a tooltip to show the ZIP code and 'diff' data
folium.GeoJson(
    geojson_data,
    name = 'hover',
    style_function=style_function,
    tooltip=folium.features.GeoJsonTooltip(fields=['ZIP', 'diff'])
).add_to(map)

# Add a layer control to the map
folium.LayerControl().add_to(map)

#Display the map in Streamlit
st.write("Difference in price between median low qual house and high qual")
folium_static(map, width=1700, height=700)