import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from io import StringIO
import numpy as np

__all__ = ['download_csv', 'combine_state_data', 'update_geo', 'q2_state_plotter']


def download_csv(url: str) -> pd.DataFrame:
    """
    Downloads .csv file from the given link and converts it to a pandas DataFrame
    :param url: URL of a .csv download
    :return: DataFrame based on url .csv
    """
    import requests

    with requests.Session() as s:
        r = s.get(url)
    df = pd.read_csv(StringIO(r.text))
    return df


def combine_state_data(state_col: str, data_col: str, data: pd.DataFrame, mainland=True) -> gpd.GeoDataFrame:
    """
    Merges any data set with the US States geospatial data

    :param state_col: US States column to merge with
    :param data_col: data column to merge with
    :param data: DataFrame
    :param mainland: 'False' to generate map including non-mainland states
    :return: GeoDataFrame of merged data
    """
    states_geo = gpd.read_file('map/states.json')
    if mainland:
        states_geo = states_geo[
            (states_geo['NAME'] != 'Alaska') & (states_geo['NAME'] != 'Hawaii') & (states_geo['NAME'] != 'Puerto Rico')]
    merged_geo = states_geo.merge(data, left_on=state_col, right_on=data_col, how='inner')
    return merged_geo


def update_geo(df: gpd.GeoDataFrame, to_drop: pd.Series, latest=False):
    """
    Convert time series data to current data (to plot on map) by updating all 'times' to latest time and removing
    old rows
    :param to_drop: column to update
    :param df: DataFrame
    :param latest: print latest date or not
    """
    if latest:
        df['date'] = pd.to_datetime(df['date'])
        print(df['date'].max())
    df = df.sort_values('date').drop_duplicates(to_drop, keep='last')
    return df


def q2_state_plotter(data: pd.DataFrame, state_name: str, axs, subplot: int):
    """
    Plots cases over time with both a linear and polynomial model trained with pre-May 25 case data
    :param subplot: indicates which subplot of the given axis to plot over
    :param axs: axis on which to plot
    :param data: DataFrame of relevant data
    :param state_name: state to plot
    :return:
    """
    # DataFrame for Minnesota cases; convert to Time Series
    mask = data['state'] == state_name
    state = data[mask]
    state.index = pd.to_datetime(state['date'])

    # Replace dates with numeric counter for use in regression model training
    state_numeric = state.assign(date=range(len(state)))

    # Linear Regression
    pre_floyd = state_numeric.loc[:'2020-05-25']
    pre_floyd_x, pre_floyd_y = pre_floyd[['date']], pre_floyd['cases']

    model = LinearRegression(fit_intercept=False)

    model.fit(pre_floyd_x, pre_floyd_y)
    pred = model.predict(state_numeric[['date']])
    pred_df = pd.DataFrame({'linear predictions': pred})
    pred_df.index = state.index

    # Polynomial Regression

    # pred[0] == 0, so omit it from the exponential fit
    # since ln(0) is undefined
    [intercept, slope] = np.polyfit(range(len(state) - 1), np.log(pred[1:]), 1)
    y = np.exp(slope) * np.exp(intercept * range(len(state)))

    exp_df = pd.DataFrame({'polynomial predictions': y})
    exp_df.index = state.index

    state['cases'].plot(ax=axs[subplot], ylim=0, label='Cases')
    pred_df.plot(ax=axs[subplot], ylim=0, label='Linear Predictions')
    exp_df.plot(ax=axs[subplot], ylim=0, label='Polynomial Predictions')
    axs[subplot].legend(loc='upper left')
    axs[subplot].set_title(state_name + ' Cases')
