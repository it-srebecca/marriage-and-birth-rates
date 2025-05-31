import sys
import os
import pandas as pd
import numpy as np
import pycountry
import pycountry_convert as pc

# Ensure data directory exists
os.makedirs("../data", exist_ok=True)

# Load data
marriage_df = pd.read_csv("data/marriage_data.csv")
fertility_df = pd.read_csv("data/fertility_data.csv")

# Rename columns for clarity
marriage_df.columns = ["Entity", "Code", "Year", "Projection", "Estimate"]
marriage_df["Year"] = marriage_df["Year"].astype(int)

# Reshape fertility data to long format
fertility_long = fertility_df.melt(
    id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
    var_name="Year", value_name="Fertility Rate"
)
fertility_long["Year"] = pd.to_numeric(fertility_long["Year"], errors='coerce')

# Add continent column
def get_continent(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        if not country:
            # Try common name
            for c in pycountry.countries:
                if country_name in [getattr(c, 'name', ''), getattr(c, 'official_name', ''), getattr(c, 'common_name', '')]:
                    country = c
                    break
        if not country:
            return np.nan
        country_alpha2 = country.alpha_2
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        return pc.convert_continent_code_to_continent_name(continent_code)
    except Exception:
        return np.nan

fertility_long["Continent"] = fertility_long["Country Name"].apply(get_continent)

# Average fertility rate per country per year
fertility_avg = fertility_long.groupby(
    ["Country Name", "Year", "Continent"]
)["Fertility Rate"].mean().reset_index()
fertility_avg = fertility_avg.rename(columns={"Country Name": "Country"})

# Filter marriage_df for relevant years and non-null Estimate
filtered_marriage_df = marriage_df[
    (marriage_df["Year"].isin(fertility_avg["Year"])) &
    (marriage_df["Estimate"].notnull())
]

# Merge with fertility_avg on both Country and Year
combined_df = pd.merge(
    filtered_marriage_df,
    fertility_avg,
    left_on=["Entity", "Year"],
    right_on=["Country", "Year"],
    how="inner"
)
combined_df.drop("Projection", axis=1, inplace=True)

# Save cleaned data
combined_df.to_csv("data/cleaned_combined_data.csv", index=False)

print("Data cleaning complete. Cleaned data saved to data/cleaned_combined_data.csv")