# Do we have a fertility problem, or a relationship problem?

[This article](https://www.ft.com/content/43e2b4f6-5ab7-4c47-b9fd-d611c36dad74) and [this article](https://www.ft.com/content/cef1c8b4-b278-425a-88b4-99d37bd4439b) in the FT contain quite provocative claims: that falling birth rates might be "due to" a falling number of couples. I enjoy contrarian claims, so I wanted to explore the relationship between marriage and fertility myself using statistical methods. I attempt to shed some light on the question: _do we have a fertility problem, or a relationship problem_? Do people only get married because they want babies? Do people to marry only after they've had a baby? Or is the opposite true: that people only have babies in marriages?

However, interpreting statistical results is difficult, and statistical methods alone cannot determine casality. What I've done here instead is take a look at the predictive ability of one of marriage or fertility for modelling the other. This may shed som light on aspects of the relationship between the two, although it will not tell the entire story, which is undoubtedly a complicated one.

## Findings

What I found seemed to contradict the FT's claims, and it was this: that **birth rates have been more useful in predicting marriage, and not the other way round**. Is this causal in any way? It may be, it may not be. The causal argument would be this: that more people are having children before marrying, and some who do not have children do not go on to marry, and this is increasing over time.

There are many non-causal explanations. One might be this: some people are choosing to have fewer children, even in marriages. And, as a seperate phenomenon, people are also increasingly staying single. The two phenomenons may be correlated due to a common factor rather than a direct causal link.

---

## Repo
This repository contains code and notebooks for analysing the relationship between marriage prevalence and fertility rates across European countries using time series methods. The project combines exploratory data analysis, forecasting, and causality testing to provide insights into how marriage and fertility trends have evolved and interact over time.

- **/data/**  
  Contains the cleaned dataset (`cleaned_combined_data.csv`) used for analysis.

The marriage data was sourced from [this Our World In Data article](https://ourworldindata.org/marriages-and-divorces#:~:text=Overall%2C%20we%20see%20a%20global,married%20or%20in%20a%20union.).
The fertility data was sourced from the [World Bank group](https://data.worldbank.org/indicator/SP.DYN.TFRT.IN?end=2023&start=1960&view=chart).

- **/notebooks/**  
  - `eda.ipynb`: Exploratory Data Analysis (EDA) of marriage and fertility trends.
  - `analysis.ipynb`: Main notebook for time series modeling, forecasting, and Granger causality analysis.

- **/src/**  
  - `modelling.py`: Core functions for data preparation, ARIMA and VAR forecasting, Granger causality testing, and model evaluation.
  - `viz.py`: Visualisation utilities for highlighting model results.

## Results Overview
- **Most European countries show a long-term decline in both marriage prevalence and fertility rates, though the timing and magnitude vary.**
- **Granger causality tests show that, in some countries, changes in fertility can help predict changes in marriage. The opposite is true but to a lesser extent.**
- **VAR models often outperform univariate models, especially in countries where marriage and fertility trends are closely linked.**
- **Country-specific results and interpretations are provided in the notebooks, highlighting demographic, social, and policy contexts.**
