import requests
import pandas as pd
from io import StringIO
import geopandas as gpd
import seaborn as sns


def download_csv(url: str) -> pd.DataFrame:
    with requests.Session() as s:
        r = s.get(url)
    df = pd.read_csv(StringIO(r.text))
    return df

# This method will combine any data set with the US states geospatial data set based on a specific column for each
# (This method can be used for making visualizations for the project)
def combine_state_data(state_col: str, data_col: str, data, mainland=True) -> gpd.GeoDataFrame:
    states_geo = gpd.read_file('map/states.json')
    if mainland:
        states_geo = states_geo[(states_geo['NAME'] != 'Alaska') & (states_geo['NAME'] != 'Hawaii') & (states_geo['NAME'] != 'Puerto Rico')]
    merged_geo = states_geo.merge(data, left_on=state_col, right_on=data_col, how='inner')
    return merged_geo


def main():
    us_data_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv'
    influenza_data_url = 'https://data.cdc.gov/api/views/ks3g-spdg/rows.csv?accessType=DOWNLOAD'
    influenza_data_state = download_csv(influenza_data_url)
    print(influenza_data_state.columns)


if __name__ == '__main__':
    main()
