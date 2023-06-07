import copy
import math
import joypy
import calendar
import datetime
import operator
import community
import matplotlib
import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import networkx as nx
from glob import glob
import geopandas as gpd
from cycler import cycler
from tqdm.auto import tqdm
from itertools import cycle
from scipy import interpolate
import astropy.units as units
from pylab import detrend_mean
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.colors as colors
from sklearn.metrics import r2_score
from sklearn.metrics.cluster import *
import scipy.cluster.hierarchy as shc
import matplotlib.patches as mpatches
from astropy.timeseries import LombScargle
from collections import Counter, defaultdict
from sklearn.preprocessing import minmax_scale
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import jaccard, cosine
from scipy.stats import linregress, pearsonr, zscore
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



def plot_clock_diff1(df_19, df_20, fname, hours):
    N = len(hours)
    ckc = "#46494c"
    hrs_20 = df_20.index.values.tolist()
    hrs_19 = df_19.index.values.tolist()

    for h_ in hours:
        if h_ not in hrs_19:
            df_19[h_] = 0.0
        if h_ not in hrs_20:
            df_20[h_] = 0.0
    
    bottom = 10
    theta, width = np.linspace(0.0, 2 * np.pi, N, endpoint=False, retstep=True)
    # print(theta)

    plt.figure(figsize = (10, 6))
    ax = plt.subplot(111, polar=True)

    bars = ax.bar(
        theta, df_19,
        width=width-0.03,
        bottom=bottom,
        color="#595959", alpha=0.55
    )
    bars = ax.bar(
        theta, df_20,
        width=width-0.03,
        bottom=bottom,
        color="#ff5d8f", alpha=0.55
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_rticks([])

    ticks = [f"{i}:00" for i in range(0, N, 1)]
    ax.set_xticklabels([])

    _ = ax

    ## Draw a "clock" icon inside of the graph
    ##lines for hands of a clock
    x1, y1 = [0, 90], [0, 0.5*bottom]
    x2, y2 = [0,0], [0, 0.5*bottom]
    plt.plot(x1, y1, x2, y2, linewidth=2.5, solid_capstyle='round', color=ckc, alpha=1)

    ##circle for clockface
    circle = plt.Circle((0, 0), 0.7*bottom, transform=ax.transData._b, linewidth=3, fill=False, color=ckc, alpha=1)
    ax.add_artist(circle)

    y0,y1 = ax.get_ylim()
    labels19 = [ "{}h".format(x) for x in np.round(df_19.values,decimals = 1)]
    labels20 = [ "{}h".format(x) for x in np.round(df_20.values,decimals = 1)]
    
    rotations = sorted(np.rad2deg(theta), reverse=True)
    rotations = [x + 30 for x in rotations]
    rects = ax.patches

    for x, bar, rotation, label19, label20 in zip(theta, rects, rotations, labels19, labels20):
#         print(x)
        offset = (bottom+bar.get_height()+8)/(y1-y0)
        lab = ax.text(0, 0, label19, transform=None, ha='center', va='center', fontsize=22, color="#595959", fontweight='bold')
        renderer = ax.figure.canvas.get_renderer()
        bbox = lab.get_window_extent(renderer=renderer)
        invb = ax.transData.inverted().transform([[0,0],[bbox.width,0] ])
        lab.set_position((x,offset+(invb[1][0]-invb[0][0])/2.*2.7 ) )
        lab.set_transform(ax.get_xaxis_transform())
        lab.set_rotation(rotation)

        offset = (bottom+bar.get_height()+5)/(y1-y0)
        lab = ax.text(0, 0, label20, transform=None, ha='center', va='center', fontsize=22, color="#ff5d8f", fontweight='bold')
        renderer = ax.figure.canvas.get_renderer()
        bbox = lab.get_window_extent(renderer=renderer)
        invb = ax.transData.inverted().transform([[0,0],[bbox.width,0] ])
        lab.set_position((x,offset+(invb[1][0]-invb[0][0])/2.*2.7 ) )
        lab.set_transform(ax.get_xaxis_transform())
        lab.set_rotation(rotation)
    
    plt.text(0, bottom*.8, "06", transform=ax.transData._b, ha='center', va='center', color=ckc, fontsize=18, fontweight='bold')
    plt.text(bottom*0.85, 0, "09", transform=ax.transData._b, ha='center', va='center', color=ckc, fontsize=18, fontweight='bold')
    plt.text(0, -bottom*.83, "12", transform=ax.transData._b, ha='center', va='center', color=ckc, fontsize=18, fontweight='bold')
    plt.text(-bottom*0.85, 0, "15", transform=ax.transData._b, ha='center', va='center', color=ckc, fontsize=18, fontweight='bold')
    plt.savefig(fname, transparent=True)

    plt.show()
    
def plot_clock_diff(df_19, df_20, fname):
    ckc = "#46494c"
    hrs_20 = df_20.index.values.tolist()
    hrs_19 = df_19.index.values.tolist()

    for h_ in range(24):
        if h_ not in hrs_19:
            df_19[h_] = 0.0
        if h_ not in hrs_20:
            df_20[h_] = 0.0
    N = 24
    bottom = 10
    theta, width = np.linspace(0.0, 2 * np.pi, N, endpoint=False, retstep=True)
    # print(theta)

    plt.figure(figsize = (10, 6))
    ax = plt.subplot(111, polar=True)

    bars = ax.bar(
        theta, df_19,
        width=width-0.03,
        bottom=bottom,
        color="#ff5d8f", alpha=1
    )
    bars = ax.bar(
        theta, df_20,
        width=width-0.03,
        bottom=bottom,
        color="grey", alpha=1
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_rticks([])

    ticks = [f"{i}:00" for i in range(0, 24, 1)]
    ax.set_xticklabels([])

    _ = ax

    ## Draw a "clock" icon inside of the graph
    ##lines for hands of a clock
    x1, y1 = [0, 90], [0, 0.5*bottom]
    x2, y2 = [0,0], [0, 0.5*bottom]
    plt.plot(x1, y1, x2, y2, linewidth=2.5, solid_capstyle='round', color=ckc, alpha=1)

    ##circle for clockface
    circle = plt.Circle((0, 0), 0.7*bottom, transform=ax.transData._b, linewidth=3, fill=False, color=ckc, alpha=1)
    ax.add_artist(circle)

    y0,y1 = ax.get_ylim()
    labels19 = np.round(df_19.values,decimals = 2)#[ "{}h".format(x) for x in np.round(df_19.values,decimals = 2)]
    labels20 = np.round(df_20.values,decimals = 2)#[ "{}h".format(x) for x in np.round(df_19.values,decimals = 2)]
    labels19 = [f"+{x}h" for x in np.round(labels19 - labels20, decimals=1)]
    
    rotations = sorted(np.rad2deg(theta), reverse=True)
    rotations = [x + 105 for x in rotations]
    rects = ax.patches

    for x, bar, rotation, label19, label20 in zip(theta, rects, rotations, labels19, labels20):
        offset = (bottom+bar.get_height()+9)/(y1-y0)
        lab = ax.text(0, 0, label19, transform=None, ha='center', va='center', fontsize=24, color="#ff5d8f", fontweight='bold')
        renderer = ax.figure.canvas.get_renderer()
        bbox = lab.get_window_extent(renderer=renderer)
        invb = ax.transData.inverted().transform([[0,0],[bbox.width,0] ])
        lab.set_position((x,offset+(invb[1][0]-invb[0][0])/2.*2.7 ) )
        lab.set_transform(ax.get_xaxis_transform())
        lab.set_rotation(rotation)
    
    plt.text(0, bottom*.8, "00", transform=ax.transData._b, ha='center', va='center', color=ckc, fontsize=14, fontweight='bold')
    plt.text(bottom*0.85, 0, "06", transform=ax.transData._b, ha='center', va='center', color=ckc, fontsize=14, fontweight='bold')
    plt.text(0, -bottom*.83, "12", transform=ax.transData._b, ha='center', va='center', color=ckc, fontsize=14, fontweight='bold')
    plt.text(-bottom*0.85, 0, "18", transform=ax.transData._b, ha='center', va='center', color=ckc, fontsize=14, fontweight='bold')
    plt.savefig(fname, transparent=True)

    plt.show()



def get_full_date(day, year):
    if not isinstance(day, int):
        date_ = datetime.datetime(year, day[1], day[0])
    else:
        date_ = datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1)
    return date_


def get_week_day(day, year):
    date_ = get_full_date(day, year)
    return int(date_.weekday())


def get_week(day, year):
    date_ = get_full_date(day, year)
    return int(date_.strftime('%V'))


def get_month(day, year):
    date_ = get_full_date(day, year)
    return int(date_.month)


def get_power(xdata, t):
    th = t * units.hour
    y_mag = xdata * units.mag
    ls = LombScargle(th, y_mag, center_data=True, fit_mean=False)
    frequency, power = ls.autopower(nyquist_factor=1, maximum_frequency=(1 / 2) / units.hour,
                                    minimum_frequency=(1 / 24) / units.hour,
                                    samples_per_peak=1)
      
    f = interpolate.interp1d(np.around(1/frequency, 1), power)
    total = np.sum(power) - f(24)
#     total = np.sum(power) - np.sum([f(x) for x in range(13, 24)]) #- f(24)
    s = float(np.sum([f(x) for x in [4, 6, 8, 12]]))/ total#(1-f(24)) #, f(24)
    return float(s)#float(np.sum(power))


def gen_power(df, period, t):
    ps = []
    for p in period:
        temp = normalize(df[p].values)
        ps.append(get_power(temp, t=t))
    return ps

def get_duration(arr):
    duration, count = [], 0
    for v in arr:
        count += 1
        duration.append(count)
        if v != 0:
            count = 0
    return duration

def get_density_groups(x):
    if x < 4.5:
        return '4.5'
    elif  4.5 <= x < 6.0:
        return 2
    elif  6.0 <= x < 7.5:
        return 3
    else:
        return 4   
    

def get_period_legend(x):
    leg_dict = {12:'Week Before L1', 13:'Week 1 of L1', 14:'Week 2 of L1' ,
                44:'Week Before L2' , 45:'Week 1 of L2' , 46:'Week 2 of L2' ,
                52:'Week Before L3'}
    return leg_dict[x]
    
    
def get_transition_groups(x):
    return 'Increment {}'.format(x)


def get_metric(data_, period=[2, 53], count_by='radius'):
    cols, uk_data, eng_data, nir_data, sct_data, wal_data = [], [], [], [], [], []
    for year in sorted(data_.year.unique()):
        for week in sorted(data_.week.unique()):
            if period[0] <= week <= period[-1]:
                df = data_[(data_.week == week) & (data_.year == year)]
                r = datetime.datetime.strptime('{}-W{}'.format(year, week) + '-1', '%Y-W%W-%w')
                cols.append(str(r.strftime('%d-%b-%y')))

                uk_data.append(np.median(df[[count_by]].values))

                nir = df[df['geo_code'].str[0] == '9']
                nir.append(df[df['geo_code'].str[0] == 'N'])
                if len(nir) > 0:
                    nir_data.append(np.median(nir[[count_by]].values))
                else:
                    nir_data.append(np.nan)

                eng = df[df['geo_code'].str[0] == 'E']
                if len(eng) > 0:
                    eng_data.append(np.median(eng[[count_by]].values))
                else:
                    eng_data.append(np.nan)

                wal = df[df['geo_code'].str[0] == 'W']
                if len(wal) > 0:
                    wal_data.append(np.median(wal[[count_by]].values))
                else:
                    wal_data.append(np.nan)

                sct = df[df['geo_code'].str[0] == 'S']
                if len(sct) > 0:
                    sct_data.append(np.median(sct[[count_by]].values))
                else:
                    sct_data.append(np.nan)

    return pd.DataFrame([uk_data, eng_data, nir_data, sct_data, wal_data], columns=cols,
                        index=['UK', 'ENG', 'NIR', 'SCT', 'WAL'])


def load_radius_data(file_path):
    files_ = glob(file_path, recursive=True)
    temp_data = [pd.read_csv(f, sep='', header=None, names=['week', 'radius', 'pings', 'geo_code']) for f in
                 sorted(files_)]
    radius_df = pd.concat(temp_data)
    region_df = pd.read_csv('files/region_geocode.csv')

    radius_df = radius_df[(radius_df.radius != '\\N') & (radius_df.radius != 'NaN')]
    radius_df = radius_df[radius_df.geo_code != '\\N']
    radius_df[['radius', 'pings']] = radius_df[['radius', 'pings']].apply(pd.to_numeric)
    radius_df['radius'] = (radius_df['radius'] / 1000)

    radius_df = radius_df.dropna()
    radius_df = radius_df.merge(region_df, on='geo_code')
    return radius_df


def load_sync_data(file_months, year):
    sync_df = pd.DataFrame()
    region_df = pd.read_csv('files/region_geocode.csv')
    
    for f, m in file_months.items():
        new_data = pd.read_csv('sync/{}/{}.csv'.format(year, f), sep=',')

        new_data = new_data[new_data.geo_code != '\\N']
        new_data = new_data.rename(columns={'_col1': 'count'})
        new_data['dt'] = new_data['date'].apply(lambda x: get_full_date(x, year))
        new_data['month'] = new_data['date'].apply(lambda x: get_month(x, year))
        new_data['week'] = new_data['date'].apply(lambda x: get_week(x, year))
        new_data['weekday'] = new_data['date'].apply(lambda x: get_week_day(x, year))
        new_data = new_data[(new_data.month == m[0]) | (new_data.month == m[-1])]

        sync_df = pd.concat([sync_df, new_data])

    sync_df = sync_df.dropna()
    sync_df = sync_df.merge(region_df, on='geo_code')
    return sync_df


def get_region(geo_code, reg):
    if geo_code[0] in ['9', 'N']:
        return 'Northern Ireland'
    
    elif geo_code[0] == 'S':
        return 'Scotland'
    
    else:
        return reg[reg.geo_code==geo_code].region_name.values[0]
    

def get_country(geo_code):
    if geo_code[0] in ['9', 'N']:
        return 'Northern Ireland'
    
    elif geo_code[0] == 'S':
        return 'Scotland'
    
    elif geo_code[0] == 'W':
        return 'Wales'
    else:
        return 'Engalnd'


def get_sync_sec(grouped_sync_econ, xaxis, year, t):
    sync_results, idxs = [], []
    xaxis_labels = [str((datetime.datetime.strptime('{}-W{}'.format(year, w) + '-1', '%Y-W%W-%w')).strftime('%d/%b'))
                    for w in xaxis]

    for s in range(1, 9):
        idxs.append('sec_{}_count'.format(s))
        df_ = grouped_sync_econ[idxs[-1]]
        sync_results.append(gen_power(df_, xaxis, t=t))
        
    df = pd.DataFrame(sync_results, columns=xaxis_labels, index=idxs)
    df[xaxis_labels] = df[xaxis_labels].astype(float)

    return df


def get_sync_uk_countries(sync_df):
    sync_results = []
    df = None
    for year in sorted(sync_df.year.unique()):
        min_ = 1
        if year == 2020:
            min_ = 2
        sync_df_f = sync_df[sync_df.year == year]
        xaxis = [x for x in sorted(sync_df_f.week.unique()) if x >= min_ and x < max(sync_df_f.week.unique())]
        xaxis_labels = [str((datetime.datetime.strptime('{}-W{}'.format(year, w) + '-1', '%Y-W%W-%w')).strftime('%d-%b-%y'))
                        for w in xaxis]

        grouped_sync_uk = sync_df.groupby(['week', 'weekday', 'hourr']).sum()
        sync_results.append(gen_power(grouped_sync_uk['count'], xaxis, t=range(24 * 7)))

        eng = sync_df[sync_df['geo_code'].str[0] == 'E']
        grouped_sync_eng = eng.groupby(['week', 'weekday', 'hourr']).sum()
        sync_results.append(gen_power(grouped_sync_eng['count'], xaxis, t=range(24 * 7)))

        nir = sync_df[sync_df['geo_code'].str[0] == '9']
        nir.append(sync_df[sync_df['geo_code'].str[0] == 'N'])
        grouped_sync_nir = nir.groupby(['week', 'weekday', 'hourr']).sum()
        sync_results.append(gen_power(grouped_sync_nir['count'], xaxis, t=range(24 * 7)))

        wal = sync_df[sync_df['geo_code'].str[0] == 'W']
        grouped_sync_wal = wal.groupby(['week', 'weekday', 'hourr']).sum()
        sync_results.append(gen_power(grouped_sync_nir['count'], xaxis, t=range(24 * 7)))

        sct = sync_df[sync_df['geo_code'].str[0] == 'S']
        grouped_sync_sct = sct.groupby(['week', 'weekday', 'hourr']).sum()
        sync_results.append(gen_power(grouped_sync_sct['count'], xaxis, t=range(24 * 7)))
#         print(xaxis_labels)
#         print(sync_results)
        
        if df is None:
            df = pd.DataFrame(sync_results, columns=xaxis_labels, index=['UK', 'ENG', 'NIR', 'WAL', 'SCT'])
        else:
    
            df = df.megrge(pd.DataFrame(sync_results, columns=xaxis_labels, index=['UK', 'ENG', 'NIR', 'WAL', 'SCT']), on)
                            
    df[xaxis_labels] = df[xaxis_labels].astype(float)
    return df


def load_income_density_socio_data(df):
    en_ecom_data = pd.read_csv('files/totalannualincome2018.csv')
    econ_data = en_ecom_data.groupby('geo_code').median().reset_index()[['geo_code', 'income']]
    df = df.merge(econ_data, on='geo_code')
    shapes_area = gpd.read_file('shapes/infuse_dist_lyr_2011.shp')
    pop_data = pd.read_csv('files/uk_pop_2018.csv')
    pop_data = pop_data[pop_data['age_group'] == 'All ages'][['geo_code', 'population']]
    shapes_area = shapes_area.merge(pop_data, on='geo_code')
    shapes_area['density'] = shapes_area.population.values / (0.000001 * shapes_area.area.values)

    econ_df = pd.read_csv('files/census_uk_2011_econ.csv')
    shapes_area = shapes_area.merge(econ_df, on='geo_code')

    return shapes_area.merge(df, on='geo_code')


def get_sync_geo_codes(sync_df, year, xaxis=range(1, 54), soc=True):
    xaxis_labels = [str((datetime.datetime.strptime('{}-W{}'.format(year, w) + '-1', '%Y-W%W-%w')).strftime('%d/%b'))
                    for w in xaxis]
    res = []
    for gc in tqdm(sorted(sync_df.geo_code.unique()), leave=False):
        sync_df = sync_df[(sync_df.week >= xaxis[0]) & (sync_df.week <= xaxis[-1])]
        temp_df = sync_df[sync_df.geo_code == gc]
        try:
            grouped_sync_gc = temp_df.groupby(['week', 'weekday', 'hourr']).sum()
            sync_val = gen_power(grouped_sync_gc['count'], xaxis, t=range(24 * 7))
            res.extend(
                zip([gc] * len(xaxis_labels), xaxis, xaxis_labels, sync_val, temp_df.groupby(['week']).sum()['count']))
        except:
            pass
    if soc:
        return load_income_density_socio_data(
            pd.DataFrame(res, columns=['geo_code', 'week', 'period', 'sync', 'count']))
    else:
        return pd.DataFrame(res, columns=['geo_code', 'week', 'period', 'sync', 'count'])


def get_radius_geo_codes(data_, year, period=[1, 54], count_by='radius', soc=True):
    df = data_[(data_.week >= period[0]) & (data_.week <= period[1])]
    temp_ = df.groupby(['geo_code', 'week']).median()
    temp_ = temp_.reset_index()

    def _aux_format_week(w):
        return str(
            (datetime.datetime.strptime('{}-W{}'.format(year, w.values[0]) + '-1', '%Y-W%W-%w')).strftime('%d/%b'))

    temp_['period'] = temp_[['week']].apply(_aux_format_week, axis=1)

    if soc:
        return load_income_density_socio_data(temp_)
    else:
        return temp_


def plot_scatter_correlation(data_frame, var1, var2, qcuts, xlabel, ylabel, qlabels, n_groups=2, log=False,
                             column='period'):
    if n_groups == 2:
        labels, colors, subplot_leg = ['Low', 'High'], ['#c3553a', '#267aac'], ['A', 'B']
    else:
        labels, colors, subplot_leg = ['Low', 'Med', 'High'], ['#c3553a', '#c060a1', '#267aac'], ['A', 'B']

    try:
        for col in data_frame[column].unique():
            f, ax = plt.subplots(1, len(qcuts), figsize=(10 * 2, 8))
            f.suptitle('{} {}'.format(column, col), fontsize=30, y=1.05)

            df = data_frame[data_frame[column] == col]

            for qcut_idx in range(len(qcuts)):
                df.insert(len(df.columns), 'qcut{}'.format(qcut_idx),
                          pd.qcut(df[qcuts[qcut_idx]].values, n_groups, labels=False), True)

            for x_ in range(n_groups):
                for qcut_idx in range(len(qcuts)):
                    plt.gca()
                    df_qcut = df[df['qcut{}'.format(qcut_idx)] == x_]
                    print('Group {}: {}, Min: {}, Max: {}'.format(qlabels[qcut_idx], labels[qcut_idx],
                                                                  min(df_qcut[qcuts[qcut_idx]].values),
                                                                  max(df_qcut[qcuts[qcut_idx]].values)))

                    regr = LinearRegression()
                    regr = regr.fit(df_qcut[var1].values.reshape(-1, 1), df_qcut[var2].values.reshape(-1, 1))
                    linear_r2 = r2_score(df_qcut[var2].values.reshape(-1, 1),
                                         regr.predict(df_qcut[var1].values.reshape(-1, 1)))
                    if linear_r2 >= 0.2:
                        sns.regplot(x=df_qcut[var1].values, y=df_qcut[var2].values, color=colors[x_], ci=None,
                                    ax=ax[qcut_idx],
                                    label='{} {}, R^2={:.2f}'.format(labels[x_], qlabels[qcut_idx], linear_r2),
                                    scatter_kws={'alpha': 0.9})
                    else:
                        sns.scatterplot(x=df_qcut[var1].values, y=df_qcut[var2].values, color=colors[x_], ci=None,
                                        ax=ax[qcut_idx],
                                        label='{} {}, R^2={:.2f}'.format(labels[x_], qlabels[qcut_idx], linear_r2),
                                        alpha=0.9, s=60)

                    ax[qcut_idx].grid(linewidth=.5)
                    ax[qcut_idx].set_xlabel(xlabel, fontsize=28)
                    ax[qcut_idx].set_ylabel(ylabel, fontsize=28)
                    ax[qcut_idx].legend(loc='lower left', fontsize=24)
                    ax[qcut_idx].tick_params(axis='both', which='major', labelsize=24)
                    ax[qcut_idx].text(-0.15, 1.05, subplot_leg[qcut_idx], transform=ax[qcut_idx].transAxes, size=30,
                                      weight='bold')
                    if log:
                        ax[qcut_idx].set_yscale('log')
                        ax[qcut_idx].set_xscale('log')
            plt.show()
    except Exception as e:
        print(e)
        pass


def get_metric_slopes(df_gc, periods, metric='radius'):
    value_periods = []
    for p in periods:
        weeks = [df_gc[df_gc.period == p[0]].week.values[0], df_gc[df_gc.period == p[1]].week.values[0]]
        for dr in zip(df_gc[df_gc.period == p[0]].geo_code.values,
                      df_gc[df_gc.period == p[0]][metric].values,
                      df_gc[df_gc.period == p[1]][metric].values):
            value_periods.append(
                [dr[0], '{}-{}'.format(p[0], p[1]), dr[1], dr[2], linregress(weeks, [dr[1], dr[2]])[0]])
    radius_slope = pd.DataFrame(value_periods,
                                columns=['geo_code', 'period', 'init_' + metric, 'final_' + metric, 'slope_' + metric])
    return load_income_density_socio_data(radius_slope)


def plot_sync_per_sec(sync_df, year=2020, xaxis=range(1, 54)):
    hr, dy = 23, 7
    cols = ['ns-sec_%d_perc' % i for i in range(1, 9)]
    econ_df = pd.read_csv('files/census_uk_2011_econ.csv')
    sync_econ_df = sync_df.merge(econ_df, on='geo_code')

    for e in range(1, 9):
        sync_econ_df['sec_{}_count'.format(e)] = sync_econ_df[cols[e - 1]] * sync_econ_df['count']

    grouped_sync_econ = sync_econ_df[sync_econ_df.hourr.isin(range(0, 23))]
    grouped_sync_econ = grouped_sync_econ.groupby(['week', 'weekday', 'hourr']).sum()
    sync_econ = get_sync_sec(grouped_sync_econ, xaxis, year=year, t=range(0, hr * dy))

    ax = sync_econ.T.plot(figsize=(16, 8), linewidth=5, marker='o', ms=15, color=['#7e7e7e', '#c3553a', '#267aac',
                                                                                  '#d6a08b', '#9bb081', '#4f4f4f',
                                                                                  '#ced9bf', '#f0dad2'])
    plt.axvline(x=6, linestyle='dashed', lw=3, color='grey', label='Partial lockdown')
    plt.title('Mobility Synchronicity per SEC', fontsize=28)
    plt.ylabel('Mobility Synchronicity', fontsize=22)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)

    plt.grid()


def plot_corr_before_after_lockdown(corr_geo_codes, column='r2_score'):
    shapes_area = gpd.read_file('shapes/infuse_dist_lyr_2011.shp')
    f, ax = plt.subplots(1, 2, figsize=(8 * 2, 16))

    shape_corr_geo_codes_b = shapes_area.merge(corr_geo_codes[(corr_geo_codes.period == 'Before Lockdown')],
                                               on='geo_code')
    shape_corr_geo_codes_a = shapes_area.merge(corr_geo_codes[(corr_geo_codes.period == 'After Lockdown')],
                                               on='geo_code')

    shape_corr_geo_codes_b.plot(ax=ax[0], column=column, cmap=sns.diverging_palette(235, 20, as_cmap=True),
                                legend=False,
                                linewidth=0.1, edgecolor="black", vmin=-1, vmax=1)
    ax[0].text(0.22, 1.05, 'Before Lockdown', transform=ax[0].transAxes, size=30, weight='bold')

    shape_corr_geo_codes_a.plot(ax=ax[1], column=column, cmap=sns.diverging_palette(235, 20, as_cmap=True),
                                legend=False,
                                linewidth=0.1, edgecolor="black", vmin=-1, vmax=1)
    ax[1].text(0.22, 1.05, 'After Lockdown', transform=ax[1].transAxes, size=30, weight='bold')

    cax = f.add_axes([1, 0.3, 0.02, 0.4])
    sm = plt.cm.ScalarMappable(cmap=sns.diverging_palette(235, 20, as_cmap=True), norm=None)
    sm._A = []
    cbr = f.colorbar(sm, cax=cax)
    cbr.set_label("Correlation", fontsize=22, labelpad=20)
    cbr.ax.tick_params(labelsize=22)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.show()


def get_correlation_per_cases(cases_with_socio):
    cols = ['ns-sec_1_perc', 'ns-sec_2_perc', 'ns-sec_3_perc', 'ns-sec_4_perc', 'ns-sec_5_perc', 'ns-sec_6_perc',
            'ns-sec_7_perc',
            'ns-sec_8_perc', 'ns-sec_15_perc', 'white_perc', 'black_perc', 'asian_perc', 'mixed_perc', 'other_perc',
            'income', 'cases']
    correlations_dict = {}
    for col in cols:
        correlations_dict[col] = []

    for d in cases_with_socio.date.unique():
        try:
            df_ = cases_with_socio[cases_with_socio.date == d]
            corr = df_[cols].corr()
            fig = plt.figure(figsize=(10, 10))
            ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(110, 25, n=50), square=True,
                             annot=False)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            plt.title(d)
            plt.show()

            for col in cols:
                temp_list = correlations_dict[col]
                temp_list.append(corr[col]['cases'])
                correlations_dict[col] = temp_list
        except:
            for col in cols:
                temp_list = correlations_dict[col]
                temp_list.append(np.nan)
                correlations_dict[col] = temp_list

    for col in cols:
        temp_list = correlations_dict[col]
        temp_list.reverse()
        correlations_dict[col] = temp_list

    return correlations_dict


def get_correlation(df, refs):
    cols = df.select_dtypes('number').columns
    correlations_dict, res = {}, {}
    for col in cols:
        correlations_dict[col] = []
    for ref in refs:
        res[ref] = copy.deepcopy(correlations_dict)

    for p in df.period.unique():
        df_ = df[df.period == p]
        corr = df_[cols].corr()
        fig = plt.figure(figsize=(10, 10))
        ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(110, 25, n=50),
                         square=True, annot=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title(p)
        plt.show()

        for ref in refs:
            for col in cols:
                temp_list = res[ref][col]
                temp_list.append(corr[col][ref])
                res[ref][col] = temp_list

    return res


def plot_correlation_evolution(data, labels, title):
    colors = ['black', '#7e7e7e', '#c3553a', '#267aac', '#d6a08b', '#9bb081', '#4f4f4f', '#ced9bf', '#f0dad2']
    f, ax = plt.subplots(figsize=(18, 6))
    for idx in range(len(labels)):
        plt.plot(data[idx], c=colors[idx], linewidth=4, label=labels[idx])

    plt.axvline(x=39, linestyle='dashed', lw=3, color='grey', label='Partial Lockdown')

    plt.title(title, fontsize=28)
    ax.set_ylabel('Correlation', fontsize=22)
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=4)
    locs, labels = plt.xticks()
    ax.set_xticklabels([data[-1][int(idx)] for idx in locs if idx < 243], rotation=45)
    plt.grid(linewidth=.25)
    plt.show()


def plot_correlation_evolution1(data, labels, title):
    colors = ['#c3553a', '#267aac', '#95ab7a', '#d6a08b', '#400406'] 
    colors = ['#ff6d00', '#ff7900', '#9d4edd', '#7b2cbf'] #'#463f3a', 
    markers = ['o', 'v', 's', 'd', '*']
    f, ax = plt.subplots(figsize=(18, 6))
    for idx in range(len(labels)):
        plt.plot(data[idx], c=colors[idx], linewidth=4, label=labels[idx], marker=markers[idx], ms=12)

    plt.axhline(y=0, linestyle='dashed', lw=3, color='grey')

    plt.title(title, fontsize=28)
    ax.set_ylabel('Correlation', fontsize=22)
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=5)
    locs, labels = plt.xticks()
    ax.set_xticklabels([data[-1][int(idx)] for idx in locs if idx < 7], rotation=45)
    plt.grid(linewidth=.25)
    plt.show()


def get_polyfit(data, deg=50):
    x = range(len(data))
    p = np.polyfit(x, data, deg=50)
  
    return np.polyval(p, x)
    
    
def moving_avg(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_month_and_week_of_month(df, week_of_year):
    month = int(np.mean(df[df.week == week_of_year].month.values))
    wks_ = sorted(df[df.month == month].week.unique())
    return calendar.month_name[month], wks_.index(week_of_year) + 1


def aux_func(x):
    if x < 0:
        return -100 * (x)
    else:
        return -100 * (x)
    
    
def normalize(data):
    variance = np.var(data)
    data = (data - np.mean(data)) / (np.sqrt(variance))
    return data


def plot_mobility_activity(df, name, c, col="radious_perc", min_max=[-100, 100], leg="Mobility Activity (% of the baseline)"):
    uk_la_shape = gpd.read_file("shapes/infuse_dist_lyr_2011.shp")
    f, ax = plt.subplots(1, figsize=(8, 16))
    
    uk_la_shape.plot(ax=ax, legend=False, linewidth=0.2, edgecolor="grey", facecolor='white', hatch="//")
    

    results_joined_decrease = uk_la_shape.set_index("geo_code").join(df.set_index("geo_code"), on="geo_code")
    f_df = results_joined_decrease[results_joined_decrease[col] <= 0]
    f_df['reduction'] = f_df[col].apply(lambda x: np.abs(x))
    f_df.plot(ax=ax, column='reduction', cmap=sns.light_palette(c, as_cmap=True), legend=False, linewidth=0.1, 
              edgecolor="black", vmin=min_max[0], vmax=min_max[1], missing_kwds={"color":"grey", "edgecolor":"black", "label":"Increased Mobility"})
    
    f_df = results_joined_decrease[results_joined_decrease[col] > 0]
    f_df.plot(ax=ax, column=col, legend=False, linewidth=0.1, edgecolor="black", color='grey')
    

    cax = f.add_axes([1, 0.2, 0.03, 0.5])
    sm = plt.cm.ScalarMappable(cmap=sns.light_palette(c, as_cmap=True),
                               norm=plt.Normalize(vmin=min_max[0], vmax=min_max[1]))
    sm._A = []
    cbr = f.colorbar(sm, cax=cax)
    cbr.set_label(leg, fontsize=28, labelpad=20)
    cbr.ax.tick_params(labelsize=28)
    ax.set_axis_off()
    plt.savefig(name, transparent=True)
    plt.show()


def plot_sync_week(data, months, colors, title, ylims=None, log=False):
    fig = plt.figure(figsize=(18, 6))
    intervals = 12
    wds = np.repeat(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 24 / intervals)
    labels = ['%s %dh' % (i, j) for i, j in zip(wds, (list(range(0, 24, intervals)) * 7))]
    labels.append("Mon 0h")

    axs = fig.subplots(nrows=1, ncols=1, sharey=True)

    for idx in range(len(data)):
        axs.plot(data[idx], label="{}".format(months[idx]), color=colors[idx], linewidth=4, alpha=1.0)

    for i in range(0, 158, 24):
        axs.axvspan(i - 6, i + 6, color='grey' if i % 24 == 0 else 'C1', alpha=0.1)

    axs.set_xticks(np.arange(0, 169, intervals))
    axs.set_xticklabels(labels, rotation=45)
    axs.set_xlabel('Day of the week/Hour', fontsize=26)

    axs.set_ylabel(r'Out-of-home trips', fontsize=26)
    axs.axvspan(162, 168, color='grey', alpha=0.1, label='Night-time')

    axs.xaxis.set_tick_params(labelsize=26)
    axs.yaxis.set_tick_params(labelsize=26)
    axs.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs.yaxis.get_offset_text().set_fontsize(20)

    fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.95))
    axs.legend(fontsize=22, loc='upper center', facecolor='w', bbox_to_anchor=(0.5, 1.3), ncol=4)
#     plt.legend(fontsize=18, loc='upper center', )
    
    if ylims is not None:
        axs.set_ylim(ylims[0], ylims[1])

    if log:
        axs.set_yscale('log')
    
    plt.savefig(title)
    plt.show()


def get_la_color(geo_code):
    if geo_code.values[0][0] == 'E':
        return 'ENG'
    elif geo_code.values[0][0] == 'W':
        return 'WAL'
    elif geo_code.values[0][0] == 'S':
        return 'SCT'
    elif geo_code.values[0][0] == '9':
        return 'NIR'


def fft(data):
    n = len(data)
    X = np.fft.fft(data)
    sxx = ((X * np.conj(X)) / (n))
    f = -np.fft.fftfreq(n)[int(np.ceil(n / 2.)):]
    sxx = np.abs(sxx)
    sxx = sxx[int(np.ceil(n / 2.)):]
    return f, sxx


def levels(result, dtmin):
    dtmax = result['power'].max()
    lev = []
    for i in range(int(log2(dtmax / dtmin))):
        dtmin = dtmin * 2
        lev.append(dtmin)
    return lev


def nextpow2(i):
    n = 2
    while n < i:
        n = n * 2
    return n


def wavelet_plot(var, time, data, dtmin, result, color):
    """
    Code adapted from the Waipy Lib. More info at https://pypi.org/project/waipy/    
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))

    f, sxx = fft(data)
    ax.plot(np.log2(1 / f * result['dt']), sxx, color, label='Fourier spectrum', linewidth=5)
    ax.plot(np.log2(result['period']), result['global_ws'], color, alpha=0.5, label='Wavelet spectrum', linewidth=5)
    ax.plot(np.log2(result['period']), result['global_signif'], color=color, alpha=0.5, ls='--',
            label='95% confidence spectrum', linewidth=5)
    ax.legend(loc='upper left', fontsize=24)
    ax.set_ylim(0, 1.25 * np.max(result['global_ws']))
    ax.set_ylabel('Power', fontsize=28)

    xt = range(int(np.log2(result['period'][0])), int(np.log2(result['period'][-1]) + 1))
    xticks = [float(math.pow(2, p)) for p in xt]  # make 2^periods
    ax.set_xticks(xt)
    ax.set_xticklabels(xticks)
    ax.set_xlim(xmin=(np.log2(np.min(result['period']))), xmax=(np.log2(np.max(result['period']))))
    ax.set_xlabel('Period (hours)', fontsize=28)

    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    plt.grid(linewidth=.25)
    
    
def wave_bases(mother, k, scale, param):
    """
    Code adapted from the Waipy Lib. More info at https://pypi.org/project/waipy/    
    Computes the wavelet function as a function of Fourier frequency
    used for the CWT in Fourier space (Torrence and Compo, 1998)
    -- This def is called automatically by def wavelet --

    _____________________________________________________________________
    Inputs:
    mother - a string equal to 'Morlet'
    k      - a vectorm the Fourier frequecies
    scale  - a number, the wavelet scale
    param  - the nondimensional parameter for the wavelet function

    Outputs:global_signif
    daughter       - a vector, the wavelet function
    fourier_factor - the ratio os Fourier period to scale
    coi            - a number, the cone-of-influence size at the scale
    dofmin         - a number, degrees of freedom for each point in the
                     wavelet power (Morlet = 2)

    Call function:
    daughter,fourier_factor,coi,dofmin = wave_bases(mother,k,scale,param)
    _____________________________________________________________________
    """
    n = len(k)  # length of Fourier frequencies (came from wavelet.py)
    """CAUTION : default values"""
    if (mother == 'Morlet'):  # choose the wavelet function
        param = 6  # For Morlet this is k0 (wavenumber) default is 6
        k0 = param
        # table 1 Torrence and Compo (1998)
        expnt = -pow(scale * k - k0, 2) / 2 * (k > 0)
        norm = math.sqrt(scale * k[1]) * \
            (pow(math.pi, -0.25)) * math.sqrt(len(k))
        daughter = []  # define daughter as a list

        for ex in expnt:  # for each value scale (equal to next pow of 2)
            daughter.append(norm * math.exp(ex))
        k = np.array(k)  # turn k to array
        daughter = np.array(daughter)  # transform in array
        daughter = daughter * (k > 0)  # Heaviside step function
        # scale --> Fourier
        fourier_factor = (4 * math.pi) / (k0 + math.sqrt(2 + k0 * k0))
        # cone-of- influence
        coi = fourier_factor / math.sqrt(2)
        dofmin = 2  # degrees of freedom
# ---------------------------------------------------------#
    elif (mother == 'DOG'):
        param = 2
        m = param
        expnt = -pow(scale * k, 2) / 2.0
        pws = (pow(scale * k, m))
        pws = np.array(pws)
        """CAUTION gamma(m+0.5) = 1.3293"""
        norm = math.sqrt(scale * k[1] / 1.3293) * math.sqrt(n)
        daughter = []
        for ex in expnt:
            daughter.append(-norm * pow(1j, m) * math.exp(ex))
        daughter = np.array(daughter)
        daughter = daughter[:] * pws
        fourier_factor = (2 * math.pi) / math.sqrt(m + 0.5)
        coi = fourier_factor / math.sqrt(2)
        dofmin = 1
# ---------------------------------------------------------#
    elif (mother == 'PAUL'):  # Paul Wavelet
        param = 4
        m = param
        k = np.array(k)
        expnt = -(scale * k) * (k > 0)
        norm = math.sqrt(scale * k[1]) * \
        (2 ** m / math.sqrt(m * \
                            (math.factorial(2 * m - 1)))) * math.sqrt(n)
        pws = (pow(scale * k, m))
        pws = np.array(pws)
        daughter = []
        for ex in expnt:
            daughter.append(norm * math.exp(ex))
        daughter = np.array(daughter)
        daughter = daughter[:] * pws
        daughter = daughter * (k > 0)     # Heaviside step function
        fourier_factor = 4 * math.pi / (2 * m + 1)
        coi = fourier_factor * math.sqrt(2)
        dofmin = 2
    else:
        print ('Mother must be one of MORLET,PAUL,DOG')

    return daughter, fourier_factor, coi, dofmin


def wave_signif(Y, dt, scale1, sigtest, lag1, sig1v1, dof, mother, param):
    """
    Code adapted from the Waipy Lib. More info at https://pypi.org/project/waipy/
    CAUTION : default values"""
    import scipy
    from scipy import stats

    n1 = np.size(Y)
    J1 = len(scale1) - 1
    s0 = np.min(scale1)
    dj = np.log10(scale1[1] / scale1[0]) / np.log10(2)
    """CAUTION"""
    if (n1 == 1):
        variance = Y
    else:
        variance = np.var(Y)
    """CAUTION"""
    # sig1v1 = 0.95
    if (mother == 'Morlet'):
        # get the appropriate parameters [see table2]
        param = 6
        k0 = param
        fourier_factor = float(4 * math.pi) / (k0 + np.sqrt(2 + k0 * k0))
        empir = [2, -1, -1, -1]
        if(k0 == 6):
            empir[1:4] = [0.776, 2.32, 0.6]

    if(mother == 'DOG'):
        param = 2
        k0 = param
        m = param
        fourier_factor = float(2 * math.pi / (np.sqrt(m + 0.5)))
        empir = [1, -1, -1, -1]
        if(k0 == 2):
            empir[1:4] = [3.541, 1.43, 1.4]

    if (mother == 'PAUL'):
        param = 4
        m = param
        fourier_factor = float(4 * math.pi / (2 * m + 1))
        empir = [2., -1, -1, -1]
        if (m == 4):
            empir[1:4] = [1.132, 1.17, 1.5]

    period = [e * fourier_factor for e in scale1]
    dofmin = empir[0]  # Degrees of  freedom with no smoothing
    Cdelta = empir[1]  # reconstruction factor
    gamma_fac = empir[2]  # time-decorrelation factor
    dj0 = empir[3]  # scale-decorrelation factor
    freq = [dt / p for p in period]
    fft_theor = [((1 - lag1 * lag1) / (1 - 2 * lag1 *
                                       np.cos(f * 2 * math.pi) + lag1 * lag1))
                 for f in freq]
    fft_theor = [variance * ft for ft in fft_theor]
    signif = fft_theor
    if(dof == -1):
        dof = dofmin
    """CAUTION"""
    if(sigtest == 0):
        dof = dofmin
        chisquare = scipy.special.gammaincinv(dof / 2.0, sig1v1) * 2.0 / dof
        signif = [ft * chisquare for ft in fft_theor]
    elif (sigtest == 1):
        """CAUTION: if len(dof) ==1"""
        dof = np.array(dof)
        truncate = np.where(dof < 1)
        dof[truncate] = np.ones(np.size(truncate))
        for i in range(len(scale1)):
            dof[i] = (
                dofmin * np.sqrt(1 + pow((dof[i] * dt / gamma_fac / scale1[i]),
                                         2)))
        dof = np.array(dof)  # has to be an array to use np.where
        truncate = np.where(dof < dofmin)
        # minimum DOF is dofmin
        dof[truncate] = [dofmin * n for n in np.ones(len(truncate))]
        chisquare, signif = [], []
        for a1 in range(J1 + 1):
            chisquare.append(
                scipy.special.gammaincinv(dof[a1] / 2.0, sig1v1) * 2.0 /
                dof[a1])
            signif.append(fft_theor[a1] * chisquare[a1])
    """CAUTION : missing elif(sigtest ==2)"""
    return signif, fft_theor


def wavelet(Y, dt, param, dj, s0, j1, mother):
    """
    Code adapted from the Waipy Lib. More info at https://pypi.org/project/waipy/
    Computes the wavelet continuous transform of the vector Y,
       by definition:

    W(a,b) = sum(f(t)*psi[a,b](t) dt)        a dilate/contract
    psi[a,b](t) = 1/sqrt(a) psi(t-b/a)       b displace

    Only Morlet wavelet (k0=6) is used
    The wavelet basis is normalized to have total energy = 1 at all scales

    _____________________________________________________________________
    Input:
    Y - time series
    dt - sampling rate
    mother - the mother wavelet function
    param - the mother wavelet parameter

    Output:
    ondaleta - wavelet bases at scale 10 dt
    wave - wavelet transform of Y
    period - the vector of "Fourier"periods ( in time units) that correspond
             to the scales
    scale - the vector of scale indices, given by S0*2(j*DJ), j =0 ...J1
    coi - cone of influence

    Call function:
    ondaleta, wave, period, scale, coi = wavelet(Y,dt,mother,param)
    _____________________________________________________________________

    """

    n1 = len(Y)  # time series length
    # s0 = 2 * dt  # smallest scale of the wavelet
    # dj = 0.25  # spacing between discrete scales
    # J1 = int(np.floor((np.log10(n1*dt/s0))/np.log10(2)/dj))
    J1 = int(np.floor(np.log2(n1 * dt / s0) / dj))  # J1+1 total os scales
    # print 'Nr of Scales:', J1
    # J1= 60
    # pad if necessary
    x = detrend_mean(Y)  # extract the mean of time series
    pad = 1
    if (pad == 1):
        base2 = nextpow2(n1)  # call det nextpow2
    n = base2
    """CAUTION"""
    # construct wavenumber array used in transform
    # simetric eqn 5
    # k = np.arange(n / 2)

    import math
    k_pos, k_neg = [], []
    for i in np.arange(0, int(n / 2)):
        k_pos.append(i * ((2 * math.pi) / (n * dt)))  # frequencies as in eqn5
        k_neg = k_pos[::-1]  # inversion vector
        k_neg = [e * (-1) for e in k_neg]  # negative part
        # delete the first value of k_neg = last value of k_pos
        # k_neg = k_neg[1:-1]
    print(len(k_neg), len(k_pos))
    k = np.concatenate((k_pos, k_neg), axis=0)  # vector of symmetric
    # compute fft of the padded time series
    f = np.fft.fft(x, n)
    scale = []
    for i in range(J1 + 1):
        scale.append(s0 * pow(2, (i) * dj))

    period = scale
    # print period
    wave = np.zeros((J1 + 1, n))  # define wavelet array
    wave = wave + 1j * wave  # make it complex
    # loop through scales and compute transform
    for a1 in range(J1 + 1):
        daughter, fourier_factor, coi, dofmin = wave_bases(
            mother, k, scale[a1], param)  # call wave_bases
        wave[a1, :] = np.fft.ifft(f * daughter)  # wavelet transform
        if a1 == 11:
            ondaleta = daughter
    # ondaleta = daughter
    period = np.array(period)
    period = period[:] * fourier_factor

    # cone-of-influence, differ for uneven len of timeseries:
    if (((n1) / 2.0).is_integer()) is True:
        # create mirrored array)
        mat = np.concatenate(
            (np.arange(1, int(n1 / 2)), np.arange(1, int(n1 / 2))[::-1]), axis=0)
        # insert zero at the begining of the array
        mat = np.insert(mat, 0, 0)
        mat = np.append(mat, 0)  # insert zero at the end of the array
    elif (((n1) / 2.0).is_integer()) is False:
        # create mirrored array
        mat = np.concatenate(
            (np.arange(1, int(n1 / 2) + 1), np.arange(1, int(n1 / 2))[::-1]), axis=0)
        # insert zero at the begining of the array
        mat = np.insert(mat, 0, 0)
        mat = np.append(mat, 0)  # insert zero at the end of the array
    coi = [coi * dt * m for m in mat]  # create coi matrix
    # problem with first and last entry in coi added next to lines because
    # log2 of zero is not defined and cannot be plottet later:
    coi[0] = 0.1  # coi[0] is normally 0
    coi[len(coi) - 1] = 0.1  # coi[last entry] is normally 0 too
    wave = wave[:, 0:n1]
    return ondaleta, wave, period, scale, coi, f


def plot_fourier_polar(data, ax, colour, zscore=False, std=None, ylims=None, alpha=1.0, linewidth=5):
    n, scale = len(data)*2, []
    dt, dj, s0, k0 = 1, 0.125, 2, 6
    J1 = int(np.floor(np.log2(n * dt / s0) / dj))  # J1+1 total os scales
    fourier_factor = (4 * math.pi) / (k0 + math.sqrt(2 + k0 * k0))
    
    for i in range(J1 + 1):
        scale.append(s0 * pow(2, (i) * dj))
    period = np.array(scale) * fourier_factor

    xt = sorted([1, 2, 3, np.log2(6), np.log2(12), np.log2(24)])
    if len(data) > 12:
        xt.extend([np.log2(48), np.log2(168)])
        
    xticks = []
    for p in xt:
        val = int(math.pow(2, p))
        if val <= 24: 
            xticks.append('{} hrs'.format(val))
        else:
            xticks.append('{} days'.format(int(val/24)))

    f_ = -np.fft.fftfreq(n)[int(np.ceil(n / 2.)):]
    x_vals, y_vals, N = np.log2(1 / f_), data, len(xt)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles = np.append(angles, angles[:1])
    
    values = np.interp(xt, x_vals, y_vals)
    values = np.append(values, values[:1]) 
    
    if ax:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(xticks)

        ax.plot(angles, values, color=colour, linewidth=5, alpha=alpha)

    #     ax.set_xlabel('Period', fontsize=28)
    #     ax.set_ylabel('Power', fontsize=28)
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)

        if not ylims:
            ylims = [min(data), max(data)]
        ax.set_ylim(ylims[0], ylims[1])

        ax.set_theta_zero_location('N')
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.spines['polar'].set_color("grey")
        ax.spines['polar'].set_alpha(0.3)

        if std is not None:
            stds = np.interp(xt, x_vals, std)
            stds = np.append(stds, stds[:1]) 
            ax.fill_between(angles, (values-stds), (values+stds), color=colour, alpha=0.3)

    return values[:-1], xticks
    

def plot_fourier(data, ax, colour, zscore=False, std=None, ylims=None, alpha=1.0, linewidth=5):
    n, scale = len(data)*2, []
    dt, dj, s0, k0 = 1, 0.125, 2, 6
    J1 = int(np.floor(np.log2(n * dt / s0) / dj))  # J1+1 total os scales
    fourier_factor = (4 * math.pi) / (k0 + math.sqrt(2 + k0 * k0))
    for i in range(J1 + 1):
        scale.append(s0 * pow(2, (i) * dj))
    period = np.array(scale) * fourier_factor

    xt = sorted([1, 2, 3, np.log2(6), np.log2(12), np.log2(24)])
    if len(data) > 12:
        xt.extend([np.log2(48), np.log2(168)])
        
    xticks = []
    for p in xt:
        val = int(math.pow(2, p))
        if val <= 24: 
            xticks.append('{} hrs'.format(val))
        else:
            xticks.append('{} days'.format(int(val/24)))

    f_ = -np.fft.fftfreq(n)[int(np.ceil(n / 2.)):]
    x_vals, y_vals= np.log2(1 / f_), data
    values = np.interp(xt, x_vals, y_vals)
    
    if ax:
        ax.plot(xt, values, colour, linewidth=5, alpha=alpha)
        ax.set_xlabel('Period', fontsize=28)
        ax.set_ylabel('Power', fontsize=28)
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)

        if not ylims:
            ylims = [min(data), max(data)]
        ax.set_ylim(ylims[0], ylims[1])
        ax.set_xticks(xt)
        ax.set_xticklabels(xticks, rotation=45)
        ax.grid(linewidth=.5)

        if std is not None:
            stds = np.interp(xt, x_vals, std)
            stds = np.append(stds, stds[:1]) 
            ax.fill_between(xt, (values-stds), (values+stds), color=colour, alpha=0.3)

        if zscore:
            ax.axhline(y=0,linestyle='dashed',lw=2.5,color='red', alpha=0.5)
            ax.axhline(y=1,linestyle='dashed',lw=2.5,color='black',label='+/- Standard dev.')
            ax.axhline(y=-1,linestyle='dashed',lw=2.5,color='black')

    return values, xticks


def cwt(data, dt, pad, dj, s0, j1, lag1, param, mother, name):
    """
    Code adapted from the Waipy Lib. More info at https://pypi.org/project/waipy/ 
    CONTINUOUS WAVELET TRANSFORM
    pad = 1         # pad the time series with zeroes (recommended)
    dj = 0.25       # this will do 4 sub-octaves per octave
    s0 = 2*dt       # this says start at a scale of 6 months
    j1 = 7/dj       # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.72     # lag-1 autocorrelation for red noise background
    param = 6
    mother = 'Morlet'
    """

    #from cwt.lib_wavelet import wavelet,wave_signif

    variance = np.var(data)
    n = len(data)
    # Wavelet transform
    ondaleta, wave, period, scale, coi, f = wavelet(
        data, dt, param, dj, s0, j1, mother)
    # wave = np.array(wave)
    power = (np.abs(wave) ** 2)
    # Significance levels: (variance=1 for the normalized SST)
    signif, fft_theor = wave_signif(
        1.0, dt, scale, 0, lag1, 0.95, -1, mother, param)
    ones = np.ones((len(signif), n))  # expand signif --> ones (J+1)x(n)
    sig95 = [s * ones[1] for s in signif]   # vstack signif concatenate
    sig95 = power / sig95  # where ratio > 1, power is significant
    # Global wavelet spectrum & significance levels:
    global_ws = variance * (np.sum(power.conj().transpose(), axis=0) / n)
    dof = [n - s for s in scale]
    """CAUTION - DEFAULT VALUES """
    global_signif, fft_theor = wave_signif(
        variance, dt, scale, 1, lag1, 0.95, dof, mother, param)
    # Daughter wavelet

    joint_wavelet = np.concatenate((np.fft.ifft(ondaleta)[int(np.ceil(
        n / 2.)):], np.fft.ifft(ondaleta)[int(np.ceil(n / 2.)):][::-1]), axis=0)
    imag_wavelet = np.concatenate((np.fft.ifft(ondaleta).imag[int(np.ceil(
        n / 2.)):], np.fft.ifft(ondaleta).imag[int(np.ceil(n / 2.)):][::-1]), axis=0)
    nw = np.size(joint_wavelet)  # daughter's number of points
    # admissibility condition
    mean_wavelet = np.mean(joint_wavelet.real)
    mean_wavelet = np.ones(nw) * mean_wavelet
    result = {'ondaleta': ondaleta, 'wave': wave, 'period': period,
              'scale': scale, 'coi': coi, 'power': power, 'sig95': sig95,
              'global_ws': global_ws, 'global_signif': global_signif,
              'joint_wavelet': joint_wavelet, 'imag_wavelet': imag_wavelet,
              'nw': nw, 'mean_wavelet': mean_wavelet, 'dj': dj, 'j1': j1,
              'dt': dt, 'fft': f, 'mother': mother, 'data': data, 'name': name}
    return result


def get_tiers(tiers):
    if tiers[0] == 1:
        return 1
    elif tiers[1] == 1:
        return 2
    elif tiers[2] == 1:
        return 3
    elif tiers[3] == 1:
        return 4
    else:
        return 0  

        
def get_cluster(gc, clusters):
    c = 0
    while c < len(clusters):
        if gc in clusters[c]:
            return c
        else:
            c += 1

            
def plot_clusters(df, col="radious_perc", min_max=[-100, 100], leg="Mobility Activity (% of the baseline)", n=10, groupedby='geo_code'):
    uk_la_shape = gpd.read_file("shapes/infuse_dist_lyr_2011.shp")
    uk_la_shape = uk_la_shape[uk_la_shape.geo_code.str[0] =='E']
    region_df = pd.read_csv("files/region_geocode.csv")
    # '#c3553a', '#267aac'  0:'#8faadc', 1:'#f4b183'
    colors = {0:'#c3553a', 1:'#267aac', 2:'#ffd966', 3:'#a9d18e', 4:'#cb83b1', 5:'#e36bae', 6:'#c3553a',
              7:'#267aac', 8:'#bd9354', 9:'#435560', 10:'#f3c2cb', 11:'#cee6b4', 12:'#84caa9', 13:'#856c8b'}
    n_cmap=ListedColormap([colors[c] for c in sorted(df[col].unique())])
    
    if groupedby != 'geo_code':
        uk_la_shape = uk_la_shape.set_index('geo_code').join(region_df.set_index('geo_code'), on='geo_code')
    f, ax = plt.subplots(1, figsize=(8, 16))
    
    results_joined_decrease = uk_la_shape.set_index(groupedby).join(df.set_index(groupedby), on=groupedby)
    results_joined_decrease.plot(ax=ax, column=col, cmap=n_cmap, legend=True, alpha=0.75, linewidth=0.1, edgecolor="black",
                                 categorical=True, missing_kwds={"color":"grey", "edgecolor":"black", "label":"NA"})

    ax.set_axis_off()
    plt.show()
     
        
def get_matrix(new_merged, n, method, cols, groupedby='geo_code'):
    clusters, matrix = [], []
    for c in range(n):
        clusters.append(new_merged[new_merged.clusters==c][groupedby].values)
    
    # TODO: fOLLOW THE ORDER DEFINED BY "clusters" 
    gcs = [new_merged[new_merged.clusters==c][groupedby].values for c in sorted(new_merged.clusters.unique())]
    gcs = np.concatenate(gcs)
    for gc1 in tqdm(gcs, leave=False):
        line = []
        for gc2 in gcs:
            if method == 'cosine':
                line.append(1 - cosine(new_merged[new_merged[groupedby]==gc1][cols].values[0],
                                       new_merged[new_merged[groupedby]==gc2][cols].values[0]))
            elif method == 'corr':
                line.append(pearsonr(new_merged[new_merged[groupedby]==gc1][cols].values[0],
                                     new_merged[new_merged[groupedby]==gc2][cols].values[0])[0])
        matrix.append(line)



    sns.set(style="ticks")

    #Plot adjacency matrix in toned-down black and white
    fig = plt.figure(figsize=(30, 30)) 
    im = plt.matshow(matrix, cmap='viridis',interpolation='nearest', norm=colors.LogNorm())#,extent =[0, len(clusters), len(clusters), 0] )


    colors_ = ['red']
    assert len([clusters]) == len(colors_)

    # Store the axes of the plot in a variable
    ax = plt.gca()

    # Add individual patches to the plot for each community
    for partition, color in zip([clusters], colors_):
        current_idx = 0
        for module in clusters:
            ax.add_patch(matplotlib.patches.Rectangle((current_idx-.5, current_idx-.5),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=colors_[0],
                                          linewidth="3"))
            current_idx += len(module)

    # Add labels to the plot
    plt.colorbar(im, label=method)
    plt.xlabel('Local Authority')
    plt.ylabel('Local Authority')
    plt.show()
    return clusters, matrix, new_merged[groupedby].unique() 
    

def network_cluster(matrix, method, labels):
    
    for x1 in range(len(matrix)):
        for y1 in range(len(matrix[x1])): 
            if matrix[x1][y1] < 0:
                matrix[x1][y1] = 0

    G_per =nx.from_numpy_matrix(np.matrix(matrix))
    part = community.best_partition(G_per)


    # ordering the users' index in the similarity matrix
    order_vector=[]
    for ke, va in  (Counter(part.values())).items():
        order_vector.append([ke,va])

    order_vector=sorted(order_vector,key=operator.itemgetter(1),reverse=True) 

    # Create a list containing the new order of key values
    new_ord = [ke[0] for ke in order_vector]

    # Orderi the communities by size
    part2={}
    for ke, va in part.items():
        part2[ke]=new_ord.index(va)

    # Create a list of communities found by the louvain community detection algorithm
    louvain_comms = defaultdict(list)
    for node_index, comm_id in part2.items():
        louvain_comms[comm_id].append(node_index)

    louvain_comms = louvain_comms.values()


    nodes_louvain_ordered = [node for comm in louvain_comms for node in comm]

    # Convert the network to an adjacency matrix, according to the louvain ordered nodes
    adjacency_matrix = nx.to_numpy_matrix(G_per, nodelist=nodes_louvain_ordered)

    sns.set(style="ticks")

    #Plot adjacency matrix in toned-down black and white
    fig = plt.figure(figsize=(30, 30)) 
    im = plt.matshow(adjacency_matrix, cmap='viridis',interpolation='nearest', norm=colors.LogNorm())


    # Set the partitions according to the detected communities to be plotted
    partitions=[louvain_comms]
    colors_ = ["red"]
    comms = []
    for c in louvain_comms:
        line = []
        for idx in c:
            line.append(labels[idx])
        comms.append(line)
        

    # Make sure that the number of partitions and the amount of colours in the plot
    # are the same
    assert len(partitions) == len(colors_)


    # Store the axes of the plot in a variable
    ax = plt.gca()

    # Add individual patches to the plot for each community
    for partition, color in zip(partitions, colors_):
        current_idx = 0
        for module in partition:
            ax.add_patch(matplotlib.patches.Rectangle((current_idx-.5, current_idx-.5),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=colors_[0],
                                          linewidth="3"))
            current_idx += len(module)

    # Add labels to the plot
    plt.colorbar(im, label=method)
    plt.xlabel('Local Authority')
    plt.ylabel('Local Authority')
    plt.show()
    return comms


def get_new_ru_text(curr_class):
    if curr_class < 3:
        return 'Rural'
    elif 3 < curr_class:
        return 'Urban'
    else:
        return 'Urban/Sig. Rural'
    

def get_new_ru(curr_class):
    if curr_class < 3:
        return 1
    elif 3 < curr_class:
        return 5
    else:
        return 3


def get_transition_type(tier):
    if tier[0] < tier[1]:
        return 1
    else:
        return 0
    
    
def plot_double_axis_map(df, col, leg, lim_1, lim_2, c1, c2, file_name):
    uk_la_shape = gpd.read_file("shapes/infuse_dist_lyr_2011.shp")
    uk_la_shape = uk_la_shape[uk_la_shape.geo_code.str[0] =='E']

    f, ax = plt.subplots(1, figsize=(8, 16))

    results_joined = uk_la_shape.set_index("geo_code").join(df[df[col]>0].set_index("geo_code"), on="geo_code").reset_index()
    results_joined.plot(ax=ax, column=col, cmap=c1, legend=False, linewidth=0.1, 
              edgecolor="black", categorical=False, vmin=lim_1[0], vmax=lim_1[1])
    
    results_joined = uk_la_shape.set_index("geo_code").join(df[df[col]<0].set_index("geo_code"), on="geo_code").reset_index()
    results_joined.plot(ax=ax, column=col, cmap=c2, legend=False, linewidth=0.1, 
              edgecolor="black", categorical=False, vmin=lim_2[0], vmax=lim_2[1])
    
    
    cax = f.add_axes([1, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(cmap=c1, norm=plt.Normalize(vmin=lim_1[0], vmax=lim_1[1]))
    sm._A = []
    cbr = f.colorbar(sm, cax=cax,)
    cbr.ax.tick_params(labelsize=50)
   
    ax.set_axis_off()
    cax = f.add_axes([1.3, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(cmap=c2, norm=plt.Normalize(vmin=lim_2[0], vmax=lim_2[1]))
    sm._A = []
    cbr = f.colorbar(sm, cax=cax,)
    cbr.ax.tick_params(labelsize=50) 
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name)
    plt.show()
    
    
def plot_lollipop(df_, interval, file_name):
    sns.set(font_scale=2)
    sns.set_style('white')

    f, ax = plt.subplots(1, figsize=(16, 6))
    y_range = np.arange(1, 53, interval)
    df = df_[df_.week.isin(y_range)].set_index('week')
    colors = np.where(df['2020'] <= df['2019'], '#001219', '#001219')
    symbols = np.where(df['2020'] <= df['2019'], '', '')

    plt.scatter(y_range, df['2019']*100,color='#595959', s=250, label='2019', zorder=3)
    plt.scatter(y_range, df['2020']*100, color='#ff5d8f', s=250 , label='2020', zorder=3)

    for (_, row), y, c, s in zip(df.iterrows(), y_range, colors, symbols):
        plt.annotate(f"{s}{row['change']:+.1%}", (y-1.5, (max(row["2019"], row["2020"])*100)+.5), 
                     fontsize=17, color=c, fontweight='bold')
        if np.abs(row['change']) > 0.08:
            arrow = mpatches.FancyArrowPatch((y, row["2019"]*100), (y, row["2020"]*100), mutation_scale=50, color= '#d9d9d9')
            plt.gca().add_patch(arrow)

    plt.xticks(y_range, df.index)
    plt.ylim(10, 22)
    plt.xlim(-1, max(52, y_range[-1]+1))

    plt.gcf().subplots_adjust(left=0.35)
    plt.tight_layout()

    plt.axvspan(-1, 12, color='#595959', alpha=0.1, lw=0)
    plt.axvspan(25, 39, color='#595959', alpha=0.1, lw=0)
    plt.axvspan(51, max(52, y_range[-1]+1), color='#595959', alpha=0.1, lw=0)
    plt.axvline(x=13,linestyle='solid',lw=3,color='red', alpha=0.)

    plt.xlabel("Week of the year")
    plt.ylabel("Vititis to Green areas \n (% of users)")

    plt.savefig(file_name, transparent=True)
    plt.show()

    
def plot_lollipop_ru(df_, interval, file_name):
    sns.set(font_scale=2)
    sns.set_style('white')
    y_range = np.arange(1, 53, interval)
    colors_arrow = ['#001219'] * len(y_range)
    colors_ru = ['#00a3f1', '#f5d200']
    
    cols_suffix = ['_urban', '_rural']

    f, ax = plt.subplots(1, figsize=(16, 4))
    df = df_[df_.week.isin(y_range)].set_index('week')
    
    plt.scatter(y_range, df['2019']*100,color='#595959', s=250, label='2019', zorder=3)
    
    for suf, c in zip(cols_suffix, colors_ru):
        plt.scatter(y_range, df['2020'+suf]*100, color=c, s=250 , zorder=3)

    for (_, row), y, c, in zip(df.iterrows(), y_range, colors_arrow):
        c_max = ''
        for suf in cols_suffix:
            if row["change"] > 0 and row["2020"+suf] > row["2020"+c_max]:
                    c_max = suf
            elif row["change"] < 0 and row["2020"+suf] < row["2020"+c_max]:
                    c_max = suf
                
        plt.annotate(f"{row['change'+c_max]:+.1%}", (y-1.2, (max(row["2019"], row["2020"+c_max])*100)+.75), 
                     fontsize=17, color=c, fontweight='bold')
        
       
        if np.abs(row['change'+c_max]) > 0.07:
            arrow = mpatches.FancyArrowPatch((y, row["2019"]*100), (y, row["2020"+c_max]*100), mutation_scale=50, color= '#d9d9d9')
            plt.gca().add_patch(arrow)
    
    max_y = (df[["2020_urban", "2020_rural"]].max(axis=1).max() * 100) + 2
    min_y = (df[["2020_urban", "2020_rural"]].min(axis=1).min() * 100) - 1

    plt.xticks(y_range, df.index)
    plt.ylim(min_y, max_y)
    plt.xlim(-1, max(52, y_range[-1]+1.5))

    plt.gcf().subplots_adjust(left=0.35)
    plt.tight_layout()

    plt.axvspan(-1, 12, color='#595959', alpha=0.1, lw=0)
    plt.axvspan(25, 39, color='#595959', alpha=0.1, lw=0)
    plt.axvspan(51, max(52, y_range[-1]+1.5), color='#595959', alpha=0.1, lw=0)
    plt.axvline(x=13,linestyle='solid',lw=3,color='red', alpha=0.)

    plt.xlabel("Week of the year")
    plt.ylabel("% of users")

    plt.savefig(file_name, transparent=True)
    plt.show()

    
def load_devices_per_day_with_soc():
    devices_per_day = pd.read_csv('files/devices_per_day_complete.csv')
    devices_per_day['norm'] = devices_per_day.n_devices_green / devices_per_day.n_devices
    devices_per_day['dt'] = devices_per_day['processing_date'].apply(lambda _: datetime.datetime.strptime(str(_),"%Y%m%d"))
    devices_per_day['week'] =  np.array(pd.to_numeric(devices_per_day['dt'].dt.isocalendar().week), dtype=int)
    devices_per_day['year'] = pd.to_numeric(devices_per_day["dt"].dt.strftime('%Y'))

    # Adding socioeconomic information
    devices_per_day = load_income_density_socio_data(devices_per_day)
    all_devices = pd.read_csv('files/count_devices_all.csv')
    dev_perc, prop_n_devices, pro_norm = [], [], []

    for _, row in tqdm(devices_per_day.iterrows(), leave=False):
        temp1 = row.n_devices_green/devices_per_day[devices_per_day.processing_date==row.processing_date]['n_devices_green'].values.sum()
        dev_perc.append(temp1)
    devices_per_day['dev_perc'] = dev_perc

    for _, row in tqdm(devices_per_day.iterrows(), leave=False):
        temp2 = devices_per_day[devices_per_day.geo_code==row.geo_code]['dev_perc'].mean() * all_devices[all_devices.processing_date==row.processing_date]['n_devices'].values[0]
        prop_n_devices.append(temp2)
    devices_per_day['prop_n_devices'] = prop_n_devices 
    devices_per_day['prop_norm'] = devices_per_day['n_devices_green'] / devices_per_day['prop_n_devices']

    # Adding urbanisation information
    urban_rural_df = pd.read_csv('files/RUC11_LAD11_ENG.csv')
    devices_per_day  = devices_per_day.merge(urban_rural_df[['geo_code', 'RUC11CD']], on='geo_code')
    devices_per_day.RUC11CD = devices_per_day.RUC11CD.astype(int)
    devices_per_day['NRUC11'] = devices_per_day.RUC11CD.apply(lambda x: get_new_ru_text(x))
    
    return devices_per_day
