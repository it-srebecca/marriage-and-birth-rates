
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import AutoReg, VAR
from sklearn.metrics import mean_squared_error
import numpy as np
import traceback

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    df['Year'] = df['Year'].astype(int)
    df = df.sort_values(by=['Country', 'Year'])
    return df

def arima_forecast(df, column, forecast_steps=10, order=(1, 1, 1)):
    series_raw = df[column].dropna()
    if len(series_raw) < 2:
        raise ValueError(f"Not enough data for {column}")
    base = series_raw.iloc[0]
    series_pct = (series_raw - base) / base * 100
    model = ARIMA(series_pct, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    last_year = series_pct.index[-1]
    forecast_years = list(range(last_year + 1, last_year + 1 + forecast_steps))
    return series_pct, forecast, forecast_years

def granger_test(df, cause, effect, max_lag=5):
    test_result = grangercausalitytests(df[[effect, cause]], maxlag=max_lag, verbose=False)
    results = []
    for lag in range(1, max_lag + 1):
        p_val = test_result[lag][0]['ssr_ftest'][1]
        results.append({'Lag': lag, 'P-Value': p_val, 'Significant (p<0.05)': p_val < 0.05})
    return pd.DataFrame(results)


# Cell 2: Define a helper function to run univariate and multivariate models for one country

def run_models_for_country(df_country, lags=2, train_ratio=0.8):
    """
    Given a DataFrame df_country indexed by Year with columns ['Marriage', 'Fertility'],
    fit:
      1) Univariate AutoReg on 'Marriage' only
      2) Univariate AutoReg on 'Fertility' only
      3) Multivariate VAR on ['Marriage', 'Fertility']
    Split data into train/test by train_ratio. Return a dict of MSEs and % reductions,
    or None if insufficient data or missing values/variation.
    """
    # Ensure at least 12 observations
    if df_country.shape[0] < 12:
        return None

    # Check for any missing values
    if df_country[['Marriage', 'Fertility']].isnull().any().any():
        return None

    # Check that each series has at least 3 unique values
    if df_country['Marriage'].nunique() < 3 or df_country['Fertility'].nunique() < 3:
        return None

    # Split into train/test
    n_obs = len(df_country)
    train_size = int(n_obs * train_ratio)
    train = df_country.iloc[:train_size]
    test = df_country.iloc[train_size:]

    try:
        # 1) Univariate AutoReg for Marriage
        model_uni_marriage = AutoReg(train['Marriage'], lags=lags)
        fit_uni_marriage = model_uni_marriage.fit()
        forecast_uni_marriage = fit_uni_marriage.predict(start=train_size, end=n_obs - 1)
        mse_marriage_uni = mean_squared_error(test['Marriage'], forecast_uni_marriage)

        # 2) Univariate AutoReg for Fertility
        model_uni_fert = AutoReg(train['Fertility'], lags=lags)
        fit_uni_fert = model_uni_fert.fit()
        forecast_uni_fert = fit_uni_fert.predict(start=train_size, end=n_obs - 1)
        mse_fertility_uni = mean_squared_error(test['Fertility'], forecast_uni_fert)

        # 3) Multivariate VAR on both
        model_multi = VAR(train[['Marriage', 'Fertility']])
        fit_multi = model_multi.fit(maxlags=lags)
        forecast_multi = fit_multi.forecast(train[['Marriage', 'Fertility']].values, steps=len(test))
        # forecast_multi is an array with shape (len(test), 2), column 0=Marriage, 1=Fertility
        mse_marriage_multi = mean_squared_error(test['Marriage'], forecast_multi[:, 0])
        mse_fertility_multi = mean_squared_error(test['Fertility'], forecast_multi[:, 1])

        # Compute percentage reductions
        pct_improvement_marriage = 100 * (mse_marriage_uni - mse_marriage_multi) / mse_marriage_uni
        pct_improvement_fertility = 100 * (mse_fertility_uni - mse_fertility_multi) / mse_fertility_uni

        return {
            'Marriage MSE (Univariate)': mse_marriage_uni,
            'Marriage MSE (With Fertility)': mse_marriage_multi,
            'Marriage MSE Reduction (%)': pct_improvement_marriage,
            'Fertility MSE (Univariate)': mse_fertility_uni,
            'Fertility MSE (With Marriage)': mse_fertility_multi,
            'Fertility MSE Reduction (%)': pct_improvement_fertility
        }

    except Exception as e:
        # If any model fitting or forecasting fails, skip this country
        # You could also log: traceback.print_exc()
        return None


def run_models_for_continent(df, continent='Europe', lags=2, train_ratio=0.8):
    """
    Given a DataFrame `df` with columns ['Country', 'Continent', 'Year', 'Marriage', 'Fertility'],
    filter by the specified continent, then for each unique country in that continent, call
    run_models_for_country. Collect results into a DataFrame, sorted by Marriage MSE Reduction.
    """
    results = []
    # Filter df to continent
    df_cont = df[df['Continent'] == continent]

    for country in df_cont['Country'].unique():
        df_country = df_cont[df_cont['Country'] == country].sort_values('Year')
        # Re-index by Year and select only the two target columns
        df_country2 = df_country.set_index('Year')[['Marriage', 'Fertility']].copy()

        metrics = run_models_for_country(df_country2, lags=lags, train_ratio=train_ratio)
        if metrics is not None:
            metrics['Country'] = country
            results.append(metrics)
        # else: skipped silently (you can log if desired)

    if not results:
        return pd.DataFrame()  # empty

    results_df = pd.DataFrame(results)
    # Reorder columns so Country is first
    cols = ['Country',
            'Marriage MSE (Univariate)',
            'Marriage MSE (With Fertility)',
            'Marriage MSE Reduction (%)',
            'Fertility MSE (Univariate)',
            'Fertility MSE (With Marriage)',
            'Fertility MSE Reduction (%)']
    results_df = results_df[cols]
    # Sort by Marriage MSE Reduction (descending)
    results_df = results_df.sort_values(by='Marriage MSE Reduction (%)', ascending=False).reset_index(drop=True)
    return results_df


def highlight_improvements(df):
    """
    Given a DataFrame with columns ending in 'MSE Reduction (%)', apply
    a red-yellow-green gradient on those columns between -100% and +100%.
    Also formats the numeric MSE columns to two decimal places and percentages to one decimal place.
    Returns a Styler object.
    """
    styler = df.style \
        .background_gradient(
            subset=['Marriage MSE Reduction (%)'],
            cmap='RdYlGn',
            vmin=-100, vmax=100
        ) \
        .background_gradient(
            subset=['Fertility MSE Reduction (%)'],
            cmap='RdYlGn',
            vmin=-100, vmax=100
        ) \
        .format({
            'Marriage MSE (Univariate)': '{:.2f}',
            'Marriage MSE (With Fertility)': '{:.2f}',
            'Marriage MSE Reduction (%)': '{:.1f}%',
            'Fertility MSE (Univariate)': '{:.2f}',
            'Fertility MSE (With Marriage)': '{:.2f}',
            'Fertility MSE Reduction (%)': '{:.1f}%'
        })
    return styler

