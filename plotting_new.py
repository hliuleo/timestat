# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:40:21 2016

@author: hliu
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from IO import get_existing_data
from DateEvent import EVENT_TYPES, STD_NAMES


all_details, all_stats = get_existing_data('sum.xlsx')
pth = os.getcwd()
font = FontProperties(fname=pth + os.sep + 'simsun.ttc', size=14)
SECS_PER_DAY = 86400


COLORS = ('b',
          'w',
          'k',
          'r',
          'g',
          'y')

PATTERNS = ('',
            'X',
            '/',
            '*',
            '.')

PRACTICES = EVENT_TYPES[u'闻思修'][:]
STUDIES = EVENT_TYPES[u'学习'][:]
WORKOUTS = EVENT_TYPES[u'锻炼'][:]
CAREERS = EVENT_TYPES[u'工作'][:]
RESTLESSNESS = EVENT_TYPES[u'散乱'][:]
NECE_MISCEL = EVENT_TYPES[u'必要杂事'][:]
MISCEL = EVENT_TYPES[u'杂事'][:]
MAJOR_TYPES = [u'闻思修',
               u'睡眠',
               u'学习',
               u'锻炼',
               u'工作',
               u'散乱',
               u'必要杂事',
               u'杂事',
               u'day_active',
               u'unknown_type']

bar_prop = {'color': 'b',
            'hatch': '',
            'ecolor': 'r',
            'alpha': 0.5,
            'width': 0.8}

text_prop = {'ha': 'left',
             'va': 'bottom'}


##########################################################################
#                                                                        #
#                      data preparation                                  #
##########################################################################


def prep_stat_data(start_date,
                   end_date=None,
                   selected_category=None,
                   selected_dfs=None,
                   dropna=True,
                   fdate=False):
    if end_date is None:
        end_date = start_date
    if selected_dfs is None:
        selected_dfs = all_stats[start_date: end_date]
    selected_dfs = selected_dfs.drop('raw', axis='columns')
    if selected_category:
        pass
    else:
        selected_category = selected_dfs.columns[:]
    selected_dfs = selected_dfs.reindex(columns=selected_category)
    if dropna:
        selected_dfs = selected_dfs.dropna(axis='columns', how='all')
    selected_dfs = selected_dfs.fillna(0)
    selected_dfs = selected_dfs.stack()
    selected_dfs = selected_dfs.swaplevel(0, 1).sort_index()
    selected_dfs.index.names = ['type', 'date']
    return selected_dfs


def prep_detail_data(start_date,
                     end_date=None,
                     selected_category=None,
                     selected_dfs=None,
                     fdate=False,
                     dropna=True):

    def add_missing_date(d_range, a):
        a.index = a.date
        missing_date = set(d_range) - set(a.index)
        added = []
        default = {'name': a['name'][0],
                   'type': a['type'][0],
                   'start': np.nan,
                   'end': np.nan,
                   'diff': 0,
                   'ID': 0}
        for d in missing_date:
            default['date'] = d
            default_val = [default[col] for col in a.columns]
            added.append(default_val)
        added = pd.DataFrame(added, columns=a.columns)
        added.index = missing_date
        a = a.append(added)
        a = a.sort_index()
        return a

    def fill_date(selected_dfs):
        d_range = pd.date_range(start_date, end_date)
        selected_dfs = selected_dfs.groupby(['type', 'name']).apply(lambda x: add_missing_date(d_range, x))
        selected_dfs.set_index('ID', inplace=True, append=True)
        selected_dfs.index.names = ['type', 'name', 'date', 'ID']
        selected_dfs.index = selected_dfs.index.reorder_levels(['date', 'type', 'name', 'ID'])
        selected_dfs = selected_dfs.sort_index()
        return selected_dfs

    prep_detail_data.fill_date = fill_date

    if end_date == None:
        end_date = start_date
    if selected_dfs is None:
        selected_dfs = all_details[start_date: end_date]
    if selected_category == None:
        selected_category = selected_dfs.name.unique()
    selected_dfs = selected_dfs[selected_dfs.name.isin(selected_category)]
    selected_dfs['date'] = selected_dfs.index
    selected_dfs['ID'] = range(len(selected_dfs))
    selected_dfs['diff'] = map(td_2hrs, selected_dfs.end - selected_dfs.start)
    selected_dfs = selected_dfs[['date', 'type', 'name', 'ID',
                                 'start', 'end', 'diff']]
    if not dropna:
        existing_names = selected_dfs.name.unique()
        missing_names = list(set(selected_category) - (set(selected_category) & set(existing_names)))
        if missing_names is not []:
            added = []
            d = start_date
            ID = 0
            s = np.nan
            e = np.nan
            diff = 0
            for n in missing_names:
                for t in EVENT_TYPES:
                    if n in EVENT_TYPES[t]:
                        t = t
                        break
                added.append([d, t, n, ID, s, e, diff])
        added = pd.DataFrame(added, columns=selected_dfs.columns)
        selected_dfs = selected_dfs.append(added)

    if fdate:
        selected_dfs = fill_date(selected_dfs)
    else:
        selected_dfs = selected_dfs.pivot_table(index=['date', 'type', 'name', 'ID'],
                                                values=['start', 'end', 'diff'],
                                                aggfunc=max) # lambda x:x doesn't work here coz it does not reduce.
    return selected_dfs


##########################################################################
#                                                                        #
#                      format  conversion                                #
##########################################################################

def num_2str_hrmin(hrs):
    hour = int(hrs)
    minute = int((hrs-hour)*60)
    return '%d:%02d' % (hour, minute)


def td_2hrs(x):
    if pd.isnull(x):
        return x
    else:
        return x.seconds/3600.


def time_2num(time):
    time = datetime.time(time)
    hour = time.hour
    minute = time.minute
    if hour < 5:
        hour += 24
    return hour*60 + minute


def m2hm(x, i):
    h = int(x / 60)
    m = int(x % 60)
    h = (h if h < 24 else h-24)
    return '%(h)02d:%(m)02d' % {'h': h, 'm': m}


def avg_time(times):
    avg = 0
    for elem in times:
        avg += elem.second + 60*elem.minute + 3600*elem.hour
    avg /= len(times)
    rez = str(avg/3600) + ' ' + str((avg%3600)/60) + ' ' + str(avg%60)
    return datetime.strptime(rez, "%H %M %S")


##########################################################################
#                                                                        #
#                      basic plotting                                    #
##########################################################################


text_args = {'xoff': 0,
             'yoff': 0.1,
             'fmtfunc': num_2str_hrmin,
             'notshown': [],
             'text_prop': text_prop}


def bar_plot(bar_dfs,
             yerr=None,
             bar_pos=None,
             tick_labels=None,
             tick_locs=None,
             bottom=None,
             bar_prop=bar_prop,
             text_anno=True,
             text_args=text_args):
    if bar_pos is None:
        bar_pos = np.arange(len(bar_dfs))
    if tick_labels is None:
        tick_labels = bar_dfs.index
    if tick_locs is None:
        tick_locs = bar_pos+0.5
    if bottom is None:
        bottom = np.array([0]*len(bar_pos))
    if yerr is None:
        yerr = np.array([0]*len(bar_dfs))
    plt.bar(bar_pos,
            bar_dfs.values,
            yerr=yerr,
            bottom=bottom,
            **bar_prop)
    plt.xticks(tick_locs,
               tick_labels,
               rotation='vertical',
               fontproperties=font)
    if text_anno is True:
        text(bar_pos, bar_dfs.values,
             **text_args)
    return plt.gca()


def pie_plot(pie_data):
    pie_labels = ['%.2f' % x + r'%' for x in pie_data.values]
    legend_labels = list(pie_data.index)
    patches, text = plt.pie(pie_data, labels=pie_labels)
    plt.legend(patches, legend_labels,
               prop=font, frameon=False,
               loc='2', bbox_to_anchor=(.9, .9))


def line_plot(dfs, ax, text_anno=False, legend_anno=True, fmt_date=True, text_args=text_args):

    # ax = plt.gca()
    names = dfs.index.levels[0]
    for n in names:
        currt_df = dfs.xs(n)
        dates = currt_df.index
        values = currt_df.values
        plt.plot(dates, values, 'o-', label=n)
        if text_anno is True:
            text(dates, values, **text_args)
    plt.grid(which='both')
    # plt.xticks(rotation=90)
    if fmt_date:
        plt.gcf().autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator())
    if legend_anno:
        ax.legend(prop=font,
                  bbox_to_anchor=(1.05, 0.8, .4, 0.3),
                  frameon=False,
                  mode="expand")
    return ax


def text(xlocs, yvals,
         xoff=0.5, yoff=0.1,
         fmtfunc=num_2str_hrmin,
         text_prop=text_prop,
         notshown=[]):

    text_info = zip(xlocs,
                    yvals+yoff,
                    [fmtfunc(v) for v in yvals])
    for x, y, t in text_info:
        if t not in notshown:
            plt.text(x, y, t, **text_prop)


##########################################################################
#                                                                        #
#                      functional plotting                               #
##########################################################################


def plot_types(data, ax, kind='line'):

    l_text_args = text_args.copy()

    if kind is 'pie':
        data = data.div(data.day_active, level=1)
        pie_data = data.mean(level='type')*100
        pie_data = pie_data.drop('day_active')
        pie_plot(pie_data)

    if kind is 'line':
        l_text_args['notshown'] = ['0:00']
        line_plot(data, ax, text_args=l_text_args)

    if kind is 'bar':
        bar_dfs = data.mean(level='type')
        yerr = data.std(level='type')
        bar_dfs = bar_dfs.sort_values(ascending=False)
        yerr = yerr.reindex(index=bar_dfs.index)
        bar_plot(bar_dfs, yerr.values)


def plot_events(data, ax, kind='line', by='diff'):
    l_text_args = text_args.copy()
    if by is 'diff':

        if kind is 'bar':
            bar_dfs = data.sum(level=['name', 'date']).mean(level='name')
            yerr = data.sum(level=['name', 'date']).std(level='name')
            yerr = yerr.fillna(0)
            bar_dfs = bar_dfs.sort('diff', ascending=False)
            yerr = yerr.reindex(index=bar_dfs.index)
            # yerr = [0]*len(bar_dfs)
            l_text_args['xoff'] = 0.5
            l_text_args['yoff'] = 0.1
            bar_plot(bar_dfs, yerr.values, text_args=l_text_args)

        if kind is 'pie':
            data = data.sum(level='name')
            pie_data = data/data.sum()*100
            pie_plot(pie_data)

        if kind is 'line':
            data = data['diff']
            data.index = data.index.droplevel(['type', 'ID'])
            data = data.groupby(level=['date', 'name']).sum()
            data = data.swaplevel('date', 'name').sort_index()
            l_text_args['notshown'] = ['0:00']
            line_plot(data, ax, text_args=l_text_args)

    elif by is 'occur':
        data = data['start'].apply(time_2num)
        data = data.dropna(axis='index')
        data.index = data.index.droplevel(['type', 'ID'])
        data = data.swaplevel('date', 'name').sort_index()
        l_text_args['xoff'] = 0
        l_text_args['yoff'] = -1
        l_text_args['fmtfunc'] = lambda x: m2hm(x, 0)
        line_plot(data, ax, text_args=l_text_args)
        ax.yaxis.set_major_formatter(FuncFormatter(m2hm))
        ax.invert_yaxis()


def plot_day_bar(data, ax, by='type', xloc='start'):

    def bar_plot(currt_df, n):
        #  currt_df = currt_df.reset_index().set_index(['name', 'ID'])
        locs = currt_df[xloc]
        currt_df = currt_df['diff']
        bar_data = np.array(currt_df.values)
        bar_pos = map(lambda x: x.to_datetime(), locs)
        tick_locs = map(lambda x: x+timedelta(days=0.01), bar_pos)
        index = currt_df.index.names.index('name')
        texts = currt_df.index.get_level_values(index)
        plt.bar(bar_pos, bar_data, width=0.01,
                label=n,
                color=labels[idx][0],
                hatch=labels[idx][1])

        y_d = bar_data + 0.1
        y_t = bar_data + 0.5
        xtick_locs.extend(tick_locs)
        text_info = zip(tick_locs,
                        y_d, y_t,
                        [num_2str_hrmin(v) for v in y_d],
                        texts)
        for x, y_d, y_t, t_d, t_t in text_info:
        #   plt.text(x, y_d, t_d, ha='center', fontproperties=font)
            plt.text(x, y_t, t_t, ha='center', va='bottom',
                     rotation=90, fontproperties=font,
                     color='r')

    data.index = data.index.droplevel(level='date')
    data = data.dropna()
    labels = [(c, p) for p in PATTERNS for c in COLORS][: len(data.index.levels[0])]
    xtick_locs = []

    if by is 'type':
        data = data.drop(u'睡眠', level='type')
        names = data.index.get_level_values(0).unique()
        for idx, n in enumerate(names):
            currt_df = data.xs(n, level='type')  # index include all elements from the index of data, need to reindex to solve the problem
            bar_plot(currt_df, n)

    if by is 'event':
        data.index = data.index.droplevel(level=['date', 'type'])
        names = data.index.get_level_values(1).unique()
        for n in names:
            currt_df = data.xs(n, level='name')
            bar_plot(currt_df, n)

    plt.xticks(xtick_locs, rotation=90)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.legend(frameon=False, prop=font, bbox_to_anchor=(1, 0.8, 0.4, 0.3))


def plot_comp_bar(data, ax, plan, by='type'):
    data = data.sum(level=by)
    bottom = np.where((plan-data) > 0, 0, plan)
    actual = data - bottom
    bar_plot(plan, color='r', alpha=1)
    bar_plot(actual, bottom=bottom, alpha=1)


##########################################################################
#                                                                        #
#                      summary plotting                                  #
##########################################################################


def plot_week_sum(start_date, end_date):
    fig = plt.figure(figsize=(24, 30), dpi=300)
    # ax1 = plt.subplot(321)
    # plot_events_occur(ax1, [u'醒来', u'下床'], start_date, end_date)
    # ax2 = plt.subplot(322)
    # plot_events_occur(ax2, [u'睡觉'], start_date, end_date)
    df_Major_stat = prep_stat_data(start_date, end_date, selected_category=MAJOR_TYPES[:])
    df_Practice_stat = prep_detail_data(start_date, end_date, selected_category=PRACTICES[:])
    ax3 = plt.subplot(221)
    plot_types(df_Major_stat, ax3, kind='bar')
    ax4 = plt.subplot(222)
    plot_types(df_Major_stat, ax4, kind='pie')
    ax5 = plt.subplot(223)
    plot_events(df_Practice_stat, ax5, kind='bar')
    ax6 = plt.subplot(224)
    plot_events(df_Practice_stat, ax6, kind='pie')
    plt.savefig('%s-%s part1.png' % (str(start_date.date()), str(end_date.date())),
                bbox_inches='tight')
    df_Study_stat = prep_detail_data(start_date, end_date, selected_category=STUDIES[:])
    df_Restless_stat = prep_detail_data(start_date, end_date, selected_category=RESTLESSNESS[:])
    df_Workout_stat = prep_detail_data(start_date, end_date, selected_category=WORKOUTS[:])
    fig = plt.figure(figsize=(24, 30), dpi=300)
    ax1 = plt.subplot(325)
    plot_events(df_Study_stat, ax1, kind='bar')
    ax2 = plt.subplot(326)
    plot_events(df_Study_stat, ax2, kind='pie')
    ax3 = plt.subplot(323)
    plot_events(df_Restless_stat, ax3, kind='line')
    ax4 = plt.subplot(324)
    plot_events(df_Restless_stat, ax4, kind='pie')
    #ax5 = plt.subplot(325)
    # plot_events(ax5, start_date, end_date, kind='bar', selected_category=CAREERS[:])
    #ax6 = plt.subplot(326)
    # plot_events(ax6, start_date, end_date, kind='pie', selected_category=CAREERS[:])
    ax7 = plt.subplot(321)
    plot_events(df_Workout_stat, ax7, kind='line')
    ax8 = plt.subplot(322)
    plot_events(df_Workout_stat, ax8, kind='pie')
    plt.savefig('%s-%s part2.png' % (str(start_date.date()), str(end_date.date())),
                bbox_inches='tight')


def plot_day_sum(date):
    data = prep_detail_data(date)
    data_stat = prep_stat_data(date)
    plt.figure(figsize=(30, 10), dpi=300)
    ax1 = plt.subplot(131)
    plot_day_bar(data, ax1, by='type')
    ax2 = plt.subplot(132)
    plot_types(data_stat, ax2, kind='pie')
    ax3 = plt.subplot(133)
    plot_types(data_stat, ax3, kind='bar')
    plt.savefig('%s.png' % str(date.date()),
                bbox_inches='tight')


def temp_check(date, text):
    e = DateEvent(date, text)
    df = prep_detail_data(date, selected_dfs=e.events_df)
    ax = plt.gca()
    plot_day_bar(df, ax)


def day_stat(date):
    return 0
