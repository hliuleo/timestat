# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:49:56 2016

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
               u'杂事']


def plot_events_occur(ax, event_names, start_date, end_date, point='start'):

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

    selected_dfs = all_details[start_date: end_date]
    selected_dfs = selected_dfs[selected_dfs.name.isin(event_names)]
    groups = selected_dfs.groupby(by='name')
    for name in groups.groups.keys():
        e_groups = groups.get_group(name)
        plotting_date = e_groups.index
        plotting_time = np.array(e_groups[point].apply(time_2num))
        ax.plot(plotting_date,
                 plotting_time,
                 label=name)
        text_info = zip(plotting_date,
                        plotting_time-1,
                        [m2hm(x, 0) for x in plotting_time])
        for x, y, t in text_info:
            plt.text(x, y, t, ha='left')
    plt.gcf().autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(m2hm))
    ax.invert_yaxis()
    ax.legend(prop=font,
              bbox_to_anchor=(1.05, 0.8, .4, 0.3),
              frameon=False,
              mode="expand")
    return ax    


def num_2str_hrmin(hrs):
    hour = int(hrs)
    minute = int((hrs-hour)*60)
    return '%d:%02d' % (hour, minute)


def plot_events(ax, start_date, end_date, kind='line', selected_category=None):
#    ax = plot_events.ax
    selected_dfs = all_details[start_date: end_date]
    selected_dfs = selected_dfs[selected_dfs.name.isin(selected_category)]
    if len(selected_dfs) == 0:
        ax.plot()
        return ax
    td_2hrs = lambda x: x.seconds/3600.
    selected_dfs['date'] = selected_dfs.index
    selected_dfs['diff'] = selected_dfs.end - selected_dfs.start
    selected_dfs['diff'] = map(td_2hrs, selected_dfs['diff'])
    selected_stats = pd.pivot_table(selected_dfs,
                                    values='diff',
                                    index=[u'name', 'date'])

    if kind is 'bar':
        selected_stats = selected_stats.sum(level='name')
        bar_pos = np.arange(len(selected_stats))
        plt.bar(bar_pos,
                selected_stats,
                alpha=0.5)
        plt.xticks(bar_pos+0.5,
                   selected_stats.index,
                   rotation='vertical',
                   fontproperties=font)
        text_info = zip(bar_pos+0.5,
                        selected_stats.values+.1,
                        [num_2str_hrmin(v) for v in selected_stats.values])
        for x, y, t in text_info:
            plt.text(x, y, t, ha='center')

    if kind is 'pie':
        selected_stats = selected_stats.sum(level='name')
        pie_data = selected_stats/selected_stats.sum()*100
        pie_label = ['%.2f' % x + r'%' for x in pie_data]
        patches, text = plt.pie(pie_data, labels=pie_label)
        plt.legend(patches, list(selected_stats.index),
                   prop=font, frameon=False,
                   loc='2', bbox_to_anchor=(.9, .9))
    if kind is 'line':
        index = pd.date_range(start=start_date, end=end_date)
        e_names = selected_stats.index.levels[0]
        for n in e_names:
            plot_df = selected_stats[n].reindex(index)
            plot_df = plot_df.fillna(0)
            plotting_date = plot_df.index
            plotting_time = plot_df.values
            plt.plot(plotting_date, plotting_time, label=n)
            text_info = zip(plotting_date,
                            plotting_time+0.1,
                            [num_2str_hrmin(v) for v in plotting_time])
            for x, y, t in text_info:
                if t != '0:00':
                    plt.text(x, y, t, ha='left')
        plt.gcf().autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.legend(prop=font,
                  bbox_to_anchor=(1.05, 0.8, .4, 0.3),
                  frameon=False,
                  mode="expand")
    return ax


def plot_types(ax, start_date, end_date=None, kind='line', selected_category=None):

    if end_date is None:
        end_date = start_date
    day_num = end_date.day - start_date.day + 1
    if selected_category:
        selected_category.extend([u'day_active', u'unknown_type'])
        selected_dfs = all_stats.loc[start_date: end_date, selected_category]
    else:
        selected_dfs = all_stats.loc[start_date: end_date, all_stats.columns[1:]]
    #ax = plt.gca()  # plot_types.ax
    if kind is 'pie':
        selected_dfs = selected_dfs.div(selected_dfs.day_active, axis='index')
        selected_stats = selected_dfs.sum().dropna()
        selected_stats = selected_stats.drop('day_active')
        pie_data = selected_stats/day_num*100
        pie_label = ['%.2f' % x + r'%' for x in pie_data]
        patches, text = plt.pie(pie_data, labels=pie_label)
        plt.legend(patches, list(selected_stats.index),
                   prop=font, frameon=False,
                   loc='2', bbox_to_anchor=(.9, .9))
    if kind is 'line':
        selected_dfs = selected_dfs.fillna(0)
        selected_dfs = selected_dfs.drop([u'day_active', u'unknown_type'], axis=1)
        for col in selected_dfs.columns:
            plt.plot(selected_dfs.index, selected_dfs[col], label=col)
            text_info = zip(selected_dfs.index,
                            selected_dfs[col]+0.1,
                            [num_2str_hrmin(v) for v in selected_dfs[col]])
            for x, y, t in text_info:
                plt.text(x, y, t, ha='left')
        plt.legend(prop=font,
                   bbox_to_anchor=(0., 1.25, 1., .125),
                   mode="expand",
                   ncol=3)
        plt.gcf().autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    if kind is 'bar':
        selected_stats = selected_dfs.sum().dropna()/day_num
        yerr = selected_dfs.std().dropna()
        if len(yerr) == 0:
            yerr = [0]*len(selected_stats)
        bar_pos = np.arange(len(selected_stats))
        plt.bar(bar_pos,
                selected_stats,
                alpha=0.5,
                yerr=yerr,
                ecolor='r')
        plt.xticks(bar_pos+0.5,
                   selected_stats.index,
                   rotation='vertical',
                   fontproperties=font)
        text_info = zip(bar_pos+0.5,
                        selected_stats.values+.5,
                        [num_2str_hrmin(v) for v in selected_stats.values])
        for x, y, t in text_info:
            plt.text(x, y, t, ha='center')
    return ax


def plot_bar(ax, date, xaxis='start', by='type', selected_category=None):

    td_2hrs = lambda x: x.seconds/3600.
    selected_dfs = all_details.loc[date, :]
    selected_dfs['diff'] = selected_dfs.end - selected_dfs.start
    if selected_category is None:
        selected_category = selected_dfs.groupby('type').groups.keys()
        if u'睡眠' in selected_category:
            del selected_category[selected_category.index(u'睡眠')]
    labels = [(c, p) for p in PATTERNS for c in COLORS][: len(selected_category)]
    xtick_locs = []
    if by is 'type':
        selected_dfs = selected_dfs[selected_dfs.type.isin(selected_category)]
        groups = selected_dfs.groupby(by='type')
        for idx, t in enumerate(groups.groups.keys()):
            currt_df = groups[[xaxis, 'diff', 'name']].get_group(t)
            currt_df = currt_df.dropna()
            y = np.array(map(td_2hrs, currt_df['diff']))
            x = map(lambda x: x.to_datetime(), currt_df[xaxis])
            print x, y
            plt.bar(x, y, width=0.01,
                    label=t,
                    color=labels[idx][0],
                    hatch=labels[idx][1])
            y_d = y + 0.1
            y_t = y + 0.5
            x_loc = map(lambda x: x.to_datetime()+timedelta(days=0.01), currt_df[xaxis])
            xtick_locs.extend(x_loc)
            text_info = zip(x_loc,
                            y_d, y_t,
                            [num_2str_hrmin(v) for v in y_d],
                            currt_df['name'])
            for x, y_d, y_t, t_d, t_t in text_info:
            #   plt.text(x, y_d, t_d, ha='center', fontproperties=font)
                plt.text(x, y_t, t_t, ha='center', va='bottom',
                         rotation=90, fontproperties=font,
                         color='r')                
        plt.xticks(xtick_locs, rotation=90)
        plt.legend(frameon=False, prop=font, bbox_to_anchor=(.9, 0.8, .4, 0.3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
'''    if by is 'event':
        selected_dfs = selected_dfs[selected_dfs.name.isin(selected_category)]
        groups = selected_dfs.groupby(by='name')
        for idx, t in enumerate(groups.groups.keys()):
            currt_df = groups.get_group(t)
            currt_df = currt_df.dropna()
            y = np.array(map(td_2hrs, currt_df['diff']))
            x = map(lambda x: x.to_datetime(), currt_df[xaxis])
            plt.bar(x, y, width=0.01,
                    label=t,
                    color=labels[idx][0],
                    hatch=labels[idx][1])
            y_d = y + 0.1
            y_t = y + 0.5
            x_loc = map(lambda x: x.to_datetime()+timedelta(days=0.01), currt_df[xaxis])
            xtick_locs.extend(x_loc)
            plt.xticks(xtick_locs, rotation=90)
        plt.legend(frameon=False, prop=font, bbox_to_anchor=(.9, 0.8, .4, 0.3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    return ax'''


def plot_week_sum(start_date, end_date):
    fig = plt.figure(figsize=(24, 30), dpi=300)
    ax1 = plt.subplot(321)
    plot_events_occur(ax1, [u'醒来', u'下床'], start_date, end_date)
    ax2 = plt.subplot(322)
    plot_events_occur(ax2, [u'睡觉'], start_date, end_date)
    ax3 = plt.subplot(323)
    plot_types(ax3, start_date, end_date, kind='bar', selected_category=MAJOR_TYPES[:])
    ax4 = plt.subplot(324)
    plot_types(ax4, start_date, end_date, kind='pie', selected_category=MAJOR_TYPES[:])
    ax5 = plt.subplot(325)
    plot_events(ax5, start_date, end_date, kind='bar', selected_category=PRACTICES[:])
    ax6 = plt.subplot(326)
    plot_events(ax6, start_date, end_date, kind='pie', selected_category=PRACTICES[:])
    plt.savefig('%s-%s part1.png' % (str(start_date.date()), str(end_date.date())),
                bbox_inches='tight')
    fig = plt.figure(figsize=(24, 30), dpi=300)
    ax1 = plt.subplot(427)
    plot_events(ax1, start_date, end_date, kind='bar', selected_category=STUDIES[:])
    ax2 = plt.subplot(428)
    plot_events(ax2, start_date, end_date, kind='pie', selected_category=STUDIES[:])
    ax3 = plt.subplot(423)
    plot_events(ax3, start_date, end_date, kind='line', selected_category=RESTLESSNESS[:])
    ax4 = plt.subplot(424)
    plot_events(ax4, start_date, end_date, kind='pie', selected_category=RESTLESSNESS[:])
    ax5 = plt.subplot(425)
    plot_events(ax5, start_date, end_date, kind='bar', selected_category=CAREERS[:])
    ax6 = plt.subplot(426)
    plot_events(ax6, start_date, end_date, kind='pie', selected_category=CAREERS[:])
    ax7 = plt.subplot(421)
    plot_events(ax7, start_date, end_date, kind='line', selected_category=WORKOUTS[:])
    ax8 = plt.subplot(422)
    plot_events(ax8, start_date, end_date, kind='pie', selected_category=WORKOUTS[:])
    plt.savefig('%s-%s part2.png' % (str(start_date.date()), str(end_date.date())),
                bbox_inches='tight')


def plot_day_sum(date):
    plt.figure(figsize=(30, 10), dpi=300)
    ax1 = plt.subplot(131)
    plot_bar(ax1, date, by='type')
    ax2 = plt.subplot(132)
    plot_types(ax2, date, kind='pie')
    ax3 = plt.subplot(133)
    plot_types(ax3, date, kind='bar')
    plt.savefig('%s.png' % str(date.date()),
                bbox_inches='tight')


def temp_check(e):
    all_details = e.events_df
    ax = plt.gca()
    plot_bar(ax, e.date)