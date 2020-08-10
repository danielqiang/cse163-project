import pandas as pd
import geopandas as gpd

from io import StringIO


__all__ = ['download_csv', 'combine_state_data', 'update_geo']


def download_csv(url: str) -> pd.DataFrame:
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

    :param df: DataFrame
    :param latest: print latest date or not
    :param to_drop: column to delete duplicates of
    """
    if latest:
        df['date'] = pd.to_datetime(df['date'])
        print(df['date'].max())
    df = df.sort_values('date').drop_duplicates(to_drop, keep='last')
    return df
