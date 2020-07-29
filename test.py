import requests
import pandas as pd
from io import StringIO


def download_csv(url: str) -> pd.DataFrame:
    with requests.Session() as s:
        r = s.get(url)
    df = pd.read_csv(StringIO(r.text))
    return df


def main():
    us_data_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv'
    df = download_csv(us_data_url)
    print(df.to_string())


if __name__ == '__main__':
    main()
