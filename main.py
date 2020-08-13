import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from helpers import download_csv
import datetime

_US_STATES_DATA_URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
_US_DATA_URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv'
_WORLD_DATA_URL = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
_US_CASES_BY_AGE_URL = 'https://data.sfgov.org/api/views/sunc-2t3k/rows.csv?accessType=DOWNLOAD'
_US_COMPREHENSIVE_URL = 'https://covidtracking.com/api/v1/us/daily.csv'


def q1():
    flu_cases2019 = 35520883
    flu_deaths2019 = 34157
    flu_hospitalizations2019 = 490561

    us_data = download_csv(_US_COMPREHENSIVE_URL)

    # Convert to Time Series
    us_data.index = pd.to_datetime(us_data['date'], format='%Y%m%d')
    us_data.drop(['date'], axis=1, inplace=True)

    # Replace dates with numeric counter for use in regression model training
    us_numeric = us_data.assign(date=range(len(us_data)))

    # Linear Regression
    us_numeric = us_numeric[us_numeric['hospitalizedCurrently'].notna()]
    x, y = us_numeric[['date']], us_numeric['hospitalizedCumulative']
    model = LinearRegression()
    model.fit(x, y)
    pred = model.predict(us_numeric[['date']])
    pred_df = pd.DataFrame({'linear predictions': pred})
    pred_df.index = us_numeric.index

    # print(us_numeric.to_string())

    # Plotting
    fig, ax = plt.subplots(1)
    us_data['hospitalizedCumulative'].plot(ax=ax, ylim=0, label='Hospitalized')
    pred_df.plot(ax=ax, ylim=0, label='Linear Predictions')
    ax.set_xlim([datetime.date(2020, 3, 1), datetime.date(2020, 12, 2)])
    ax.set_ylim([0, 500000])

    # Disgusting hacky way to fix this line issue
    print(model.coef_, model.intercept_)

    ax.legend(loc='upper right')
    ax.set_title('US Hospitalizations')
    fig.savefig('US Hospitalizations', bbox_inches='tight', pad_inches=0.2)


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

    # Plotting
    fig, ax = plt.subplots(1)
    minnesota['cases'].plot(ax=ax, ylim=0, label='Cases')
    pred_df.plot(ax=ax, ylim=0, label='Linear Predictions')
    exp_df.plot(ax=ax, ylim=0, label='Polynomial Predictions')
    ax.legend(loc='upper left')
    ax.set_title('Minnesota Cases')

    fig.savefig('minnesota_cases.png', bbox_inches='tight', pad_inches=0.2)


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

    # Joining/reformatting data
    us_mask = global_recoveries_data['Country/Region'] == 'US'
    global_recoveries_data = global_recoveries_data[us_mask]
    global_recoveries_data = global_recoveries_data[['Value', 'Date']]
    us_combined_data = us_data.merge(global_recoveries_data, how='inner', left_on='date', right_on='Date')
    us_combined_data.rename(columns={'Value': 'recoveries'}, inplace=True)
    us_combined_data = us_combined_data.drop(columns='Date')
    us_combined_data['recoveries'] = us_combined_data['recoveries'].astype('int64')

    us_combined_data.index = pd.to_datetime(us_combined_data['date'])
    us_combined_data.drop(['date'], axis=1, inplace=True)

    us_combined_data['recovery rate'] = (us_combined_data['recoveries'] / us_combined_data[
        'deaths'])
    us_combined_data = us_combined_data[us_combined_data['recovery rate'].notna()]

    fig, ax = plt.subplots(1)
    us_combined_data['recovery rate'].plot(ax=ax)
    ax.set_title('US Covid-19 Recovery Rate')
    plt.ylabel('Recoveries / Deaths')
    ax.set_xlim([datetime.date(2020, 2, 29), us_combined_data.index.max()])

    print(us_combined_data.to_string())
    fig.savefig('Recovery Rates.png', bbox_inches='tight', pad_inches=0.2)


def q4():
    df = download_csv(_US_CASES_BY_AGE_URL)
    # df = pd.read_csv('cases_by_age.csv')
    df['Specimen Collection Date'] = df['Specimen Collection Date'].apply(pd.to_datetime)
    df.drop(columns=['Unnamed: 0', 'New Confirmed Cases'], inplace=True, errors='ignore')

    fig, ax = plt.subplots(1)

    all_ages = df.groupby('Specimen Collection Date').sum()
    all_ages.rename(columns={'Cumulative Confirmed Cases': 'Cum. Cases (All Ages)'},
                    inplace=True)

    age_groups = ['under 18', '18-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
    for age_group in age_groups:
        mask = df['Age Group'] == age_group
        label = f'Cum. Cases ({age_group})'

        age_group_df = df[mask].merge(all_ages, on='Specimen Collection Date')
        age_group_df.rename(columns={'Cumulative Confirmed Cases': label},
                            inplace=True)
        percentages = age_group_df[label] / age_group_df['Cum. Cases (All Ages)'] * 100
        age_group_df[age_group] = percentages
        age_group_df.plot(ax=ax, x='Specimen Collection Date', y=age_group)

    ax.set_title('COVID-19 Cases (Percentage) by Age Group in San Francisco')
    ax.set_ylabel('Percentage of Total COVID-19 Cases')
    fig.savefig('COVID-19 Cases (Percentage) by Age Group in San Francisco.png', bbox_inches='tight', pad_inches=0.2)


def q5():
    # df = download_csv(_US_STATES_DATA_URL)
    df = pd.read_csv(_US_STATES_DATA_URL)
    df['date'] = df['date'].apply(pd.to_datetime)

    fig, ax = plt.subplots(1)

    case_reductions = {}
    new_cases_by_state = {}
    for state in df['state'].unique():
        mask = df['state'] == state
        state_df = df[mask].copy()
        new_cases = df[mask]['cases'].diff()
        # diff() gives NaN for first row
        new_cases.iloc[0] = state_df['cases'].iloc[0]
        state_df['new cases'] = new_cases.astype(int)

        case_reductions[state] = state_df['new cases'].diff().min()
        new_cases_by_state[state] = state_df

    top_5 = sorted(case_reductions, key=case_reductions.get)[:5]
    for state in top_5:
        state_df = new_cases_by_state[state]
        state_df.rename(columns={'new cases': state}, inplace=True)
        state_df.plot(ax=ax, x='date', y=state, legend=True)

    ax.set_title('Largest reductions in COVID-19 cases per day (Top 5)')
    ax.set_ylabel('New Cases Per Day')
    fig.savefig('Largest reductions in COVID-19 cases per day (Top 5).png',
                bbox_inches='tight', pad_inches=0.2)
    plt.show()


def main():
    q1()
    q2()
    q3()
    # q4()
    # q5()


if __name__ == '__main__':
    main()
