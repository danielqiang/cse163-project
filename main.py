import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from helpers import download_csv

_US_STATES_DATA_URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
_US_DATA_URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv'
_INFLUENZA_DATA_URL = 'https://data.cdc.gov/api/views/ks3g-spdg/rows.csv?accessType=DOWNLOAD'
_WORLD_DATA_URL = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'



def q1():
    pass


def q2():
    us_states_df = pd.read_csv('us_states_data.csv')
    # us_states_df = download_csv(_US_STATES_DATA_URL)

    # DataFrame for Minnesota cases; convert to Time Series
    minnesota_mask = us_states_df['state'] == 'Minnesota'
    minnesota = us_states_df[minnesota_mask]
    minnesota.index = pd.to_datetime(minnesota['date'])

    # Replace dates with numeric counter for use in regression model training
    minnesota_numeric = minnesota.assign(date=range(len(minnesota)))

    # Linear Regression
    pre_floyd = minnesota_numeric.loc[:'2020-05-25']
    pre_floyd_x, pre_floyd_y = pre_floyd[['date']], pre_floyd['cases']

    model = LinearRegression(fit_intercept=False)

    model.fit(pre_floyd_x, pre_floyd_y)
    pred = model.predict(minnesota_numeric[['date']])
    pred_df = pd.DataFrame({'linear predictions': pred})
    pred_df.index = minnesota.index

    # Polynomial Regression

    # TODO: Use LinearRegression() instead of np.polyfit()
    # model = LinearRegression()
    # model.fit(pd.DataFrame({'date': range(1, len(minnesota))}), np.log(pred[1:]))
    # y = np.exp(model.coef_) * np.exp(model.intercept_ * range(len(minnesota)))

    # pred[0] == 0, so omit it from the exponential fit
    # since ln(0) is undefined
    [intercept, slope] = np.polyfit(range(len(minnesota) - 1), np.log(pred[1:]), 1)
    y = np.exp(slope) * np.exp(intercept * range(len(minnesota)))

    exp_df = pd.DataFrame({'polynomial predictions': y})
    exp_df.index = minnesota.index

    fig, ax = plt.subplots(1)
    minnesota['cases'].plot(ax=ax, ylim=0)
    pred_df.plot(ax=ax, ylim=0)
    exp_df.plot(ax=ax, ylim=0)
    ax.set_title('Minnesota Cases')

    fig.savefig('minnesota_cases.png')


def q3():
    _GLOBAL_RECOVERIES_URL = 'https://data.humdata.org/hxlproxy/data/download' \
                             '/time_series_covid19_recovered_global_narrow.csv?dest=data_edit&filter01=merge&merge' \
                             '-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX' \
                             '-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub' \
                             '%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge' \
                             '-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C' \
                             '%23region%2Bintermediate%2Bcode&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google' \
                             '.com%2Fspreadsheets%2Fd%2Fe%2F2PACX' \
                             '-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub' \
                             '%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge' \
                             '-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C' \
                             '%23region%2Bintermediate%2Bcode&merge-replace02=on&merge-overwrite02=on&filter03' \
                             '=explode&explode-header-att03=date&explode-value-att03=value&filter04=rename&rename' \
                             '-oldtag04=%23affected%2Bdate&rename-newtag04=%23date&rename-header04=Date&filter05' \
                             '=rename&rename-oldtag05=%23affected%2Bvalue&rename-newtag05=%23affected%2Binfected' \
                             '%2Bvalue%2Bnum&rename-header05=Value&filter06=clean&clean-date-tags06=%23date&filter07' \
                             '=sort&sort-tags07=%23date&sort-reverse07=on&filter08=sort&sort-tags08=%23country%2Bname' \
                             '%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01' \
                             '-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion' \
                             '&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger' \
                             '-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw' \
                             '.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data' \
                             '%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv'
    # global_recoveries_data = download_csv(_GLOBAL_RECOVERIES_URL)
    us_data = download_csv(_US_DATA_URL)

    global_recoveries_data = pd.read_csv('time_series_covid19_recovered_global_narrow.csv')

    # Reformatting/merging data
    us_mask = global_recoveries_data['Country/Region'] == 'US'
    global_recoveries_data = global_recoveries_data[us_mask]
    global_recoveries_data = global_recoveries_data[['Value', 'Date']]
    us_combined_data = us_data.merge(global_recoveries_data, how='inner', left_on='date', right_on='Date')
    us_combined_data.rename(columns={'Value': 'recoveries'}, inplace=True)
    us_combined_data = us_combined_data.drop(columns='Date')
    us_combined_data['recoveries'] = us_combined_data['recoveries'].astype('int64')
    us_combined_data['new cases'] = us_combined_data['cases'].diff()
    us_combined_data['new recoveries'] = us_combined_data['recoveries'].diff()
    us_combined_data['new deaths'] = us_combined_data['deaths'].diff()

    fig, ax = plt.subplots(1)

    print(us_combined_data.to_string())


def q4():
    pass


def q5():
    # us_state_data = download_csv(_US_STATES_DATA_URL)
    us_state_data = pd.read_csv(_US_STATES_DATA_URL)
    us_state_data['new cases'] = us_state_data['cases'].diff()




def main():
    q1()
    q2()
    q3()
    q4()
    # q5()


if __name__ == '__main__':
    main()
