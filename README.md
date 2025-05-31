# Marriage and Birth Rates in Europe: Time Series Analysis

This repository contains code and notebooks for analysing the relationship between marriage prevalence and fertility rates across European countries using time series methods. The project combines exploratory data analysis, forecasting, and causality testing to provide insights into how marriage and fertility trends have evolved and interact over time.

---

## Repository Structure

- **/data/**  
  Contains the cleaned dataset (`cleaned_combined_data.csv`) used for analysis.

The marriage data was sourced from [this Our World In Data article](https://ourworldindata.org/marriages-and-divorces#:~:text=Overall%2C%20we%20see%20a%20global,married%20or%20in%20a%20union.).
The fertility data was sourced from the [World Bank group](https://data.worldbank.org/indicator/SP.DYN.TFRT.IN?end=2023&start=1960&view=chart).

- **/notebooks/**  
  - `eda.ipynb`: Exploratory Data Analysis (EDA) of marriage and fertility trends.
  - `analysis.ipynb`: Main notebook for time series modeling, forecasting, and Granger causality analysis.

- **/src/**  
  - `modelling.py`: Core functions for data preparation, ARIMA and VAR forecasting, Granger causality testing, and model evaluation.
  - `viz.py`: Visualization utilities for highlighting model results.

---

## Key Features

- **Exploratory Data Analysis (EDA):**  
  Visualizes historical trends in marriage prevalence and fertility rates for each European country, highlighting both parallel and divergent patterns.

- **Time Series Forecasting:**  
  - **ARIMA Models:** Univariate forecasts for marriage and fertility rates.
  - **VAR Models:** Multivariate forecasts capturing dynamic interdependencies between marriage and fertility.

- **Granger Causality Testing:**  
  Assesses whether changes in one variable (e.g., fertility) can help predict changes in the other (e.g., marriage), and vice versa.

- **Model Performance Comparison:**  
  Quantifies the benefit of multivariate (VAR) modeling over univariate approaches, with country-level summaries and visualizations.

## Results Overview
- **Most European countries show a long-term decline in both marriage prevalence and fertility rates, though the timing and magnitude vary.**
- **Granger causality tests reveal that, in some countries, changes in fertility can help predict changes in marriage, and vice versa.**
- **VAR models often outperform univariate models, especially in countries where marriage and fertility trends are closely linked.**
- **Country-specific results and interpretations are provided in the notebooks, highlighting demographic, social, and policy contexts.**

---

## License

This project is for academic and research purposes. Please cite appropriately if you use or adapt the code or findings.

---

## Contact

For questions or collaboration, please open an issue or contact the repository maintainer.
