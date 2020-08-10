import requests
import pandas as pd
from io import StringIO
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def download_csv(url: str) -> pd.DataFrame:
    with requests.Session() as s:
        r = s.get(url)
    df = pd.read_csv(StringIO(r.text))
    return df


# This method will combine any data set with the US states geospatial data set based on a specific column for each
# (This method can be used for making visualizations for the project)
def combine_state_data(state_col: str, data_col: str, data, mainland=True) -> gpd.GeoDataFrame:
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


def main():
    us_states_data_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
    influenza_data_url = 'https://data.cdc.gov/api/views/ks3g-spdg/rows.csv?accessType=DOWNLOAD'

    # us_states_data = download_csv(us_states_data_url)
    us_states_df = pd.read_csv('us_states_data.csv')

    # print(us_states_data.to_string())

    # Generate graphs for question 2 (Minnesota)
    minnesota_mask = us_states_df['state'] == 'Minnesota'
    minnesota = us_states_df[minnesota_mask]
    minnesota.index = pd.to_datetime(minnesota['date'])
    # print(minnesota)
    # minnesota['cases'].plot()
    # plt.title('Minnesota Cases')
    # plt.savefig('minnesota_cases.png')

    # # Training set

    # Replace dates with numeric counter for use in regression model training
    minnesota = minnesota.assign(date=range(len(minnesota)))

    pre_floyd = minnesota.loc[:'2020-05-25']
    post_floyd = minnesota.loc['2020-05-26':]
    pre_floyd_x, pre_floyd_y = pre_floyd['date'], pre_floyd['cases']
    post_floyd_x, post_floyd_y = post_floyd['date'], post_floyd['cases']

    model = LinearRegression()
    model.fit(pre_floyd_x, pre_floyd_y)
    pred = model.predict(post_floyd_x)
    print(pred)

    # print(pre_floyd)
    # print('=' * 50)
    # print(post_floyd)



if __name__ == '__main__':
    main()
