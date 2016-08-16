# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:00:17 2016

@author: hliu
"""

from IO import *
from plotting_new import *

prep_prop = {'dropna': True,
             'selected_dfs': None,
             'fdate': True}
plot_prop = {'text_anno': False}

def sum_plot(start_date, end_date,
             selected_category=None,
             of='type',
             by='diff',
             kind='line',
             prep_prop=prep_prop):
    ax = plt.gca()
    title = str(start_date.date()) + '-' + str(end_date.date())
    def do():
        if of is 'type':
            df = prep_stat_data(start_date, end_date, selected_category=selected_category, **prep_prop)
            plot_types(df, ax, kind=kind)
        elif of is 'event':
            df = prep_detail_data(start_date, end_date, selected_category=selected_category, **prep_prop)
            plot_events(df, ax, kind=kind, by=by)
    if selected_category is None:
        ax = do()
    else:
        n = selected_category[0]
        if n in EVENT_TYPES:
            of = 'type'
        elif n in STD_NAMES:
            of = 'event'
        ax = do()
    plt.title(title)
    return ax


def comp_plot(period_dfs, kind='line'):

    ax = plt.gca()

    if kind is 'line':
        for df in period_dfs.values():
            ax = line_plot(df, ax, text_anno=False)
        major_locs = ax.xaxis.get_majorticklocs()
        all_locs = ax.xaxis.get_minorticklocs()
        day_index = np.array([np.where(all_locs==m)[0][0] for m in major_locs])
        day_lable = ['Day '+str(d+1) for d in day_index]
        plt.xticks(major_locs, day_lable, rotation=90)
        plt.grid(which='both')
        l = plt.legend(prop=font,
                       bbox_to_anchor=(1.05, 0.8, .8, 0.3),
                       frameon=False,
                       mode="expand")
        for idx, key in enumerate(period_dfs.keys()):
            l.get_texts()[idx].set_text(key)


def temp_check(date, text):
    e = DateEvent(date, text)
    df = prep_detail_data(date, selected_dfs=e.events_df)
    ax = plt.gca()
    plot_day_bar(df, ax)