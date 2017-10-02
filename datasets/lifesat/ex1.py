# example 1 handson
# author: ronny

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
oecd_bli = oecd_bli.pivot(index = "Country", columns = "Indicator", values = "Value")

# eyeball the data.
oecd_bli.head(2)
oecd_bli["Life satisfication"].head(2)

gdp_per_capita = pd.read_csv("gdp_per_capita.csv", 
    thousands = ",", delimiter = "\t", encoding = "latin", na_values = "n/a")
gdp_per_capita.rename(columns={"2015", "GDP per capita"}, inplace = True)
gdp_per_capita.set_index("Country", inplace = True)

# eyeball the data.
gdp_per_capita.head(2)

full_country_stats = pd.merge(
    left = oecd_bli, right = gdp_per_capita, left_index = True, right_index = True)
full_country_stats.sort_values(by = "GDP per capita", inplace = True)
full_country_stats
full_country_stats[["GDP per capita", 'Life satisfaction']].loc["United States"]

remove_indices = [0, 1, 6, 8, 33, 34, 35]
keep_indices = list(set(range(36)) - set(remove_indices))

sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
missing_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]

sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.axis([0, 60000, 0, 10])
position_text = {
    "Hungary" : (5000, 1),
    "Korea" : (18000, 1.7),
    "France" : (29000,2.4),
    "Australia" : (40000, 3.0),
    "United States" : (52000, 3.8),
}

for country, pos_text in position_text.items():
    pos_x, pos_y = sample_data.loc[country]
    country = "U.S." if country == "United States" else country
    plt.annotate(country, xy = (pos_x, pos_y), xytext = pos_text,
        arrowprops = dict(facecolor = "black", width = 0.5, shrink = 0.1, headwidth = 5))
    plt.plot(pos_x, pos_y, "ro")

plt.show()
