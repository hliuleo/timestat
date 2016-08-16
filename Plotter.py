# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:26:18 2016

@author: hliu
"""

class Plotter:

    def __init__(self):
        return 0

    def sum_plot(self):
        return 0

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
    
    
    def line_plot(dfs, ax, text_anno=True, legend_anno=True, fmt_date=True, text_args=text_args):
    
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