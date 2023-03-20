# Analysis of King County Housing Data


This repository contains the code and data used for analyzing King County housing data from 2021 to 2022. The goal of this analysis is to make recommendations for a King County based contractor to identify where to flip houses.

Data

The data used for this analysis was obtained from https://data.kingcounty.gov/. The main dataset includes information on 30,155 properties sold in King County between Jun 2021 and Jun 2022. The data includes a variety of features such as the price of the property, the number of bedrooms and bathrooms, the square footage of the property, and the location of the property. Zip code data and geographic boundry data were also obtained.

Code

The code used for this analysis is contained in the final_notebook.ipynb Jupyter notebook. The notebook includes all the code used for data cleaning, feature engineering, modeling, and visualization. In the data cleaning and preprocessing section, we addressed issues such as missing or inaccurate values, outliers, and duplicated data. In the exploratory data analysis section, we looked at the relationships between the different features and how they relate to the target variable (price). In the feature engineering section, we created new features from the existing ones. In the modeling section, we used several regression models to predict the price of a house given its features.

Recommendations

After evaluting the properties and classifying them based on maintenance condition and design grade, we were able to identify the top target zip codes. Based on our analysis, we recommend that the contractor focus on properties in the following zip codes: 98039, 98004, 98112, 98077, and 98109. Zip_select_map.py is a tool that can be run by Streamlit to compare all zip codes within the county. We also recommend focusing on properties with a 'class' value of 0, as these properties will gain the most value by being upgraded to class 2 properties. We were also able to identify that the contractor should expect to increase a property's value by ~$417 per square foot added.

Conclusion

presentation.housing.pdf contains an overview presentation of our findings. Our analysis provides valuable insights into the King County housing market and can help the contractor make informed decisions about where to invest.
