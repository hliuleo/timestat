# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:21:51 2016

@author: hliu
"""


class Preparer:

    def __init__(self):
        return 0

    def prep_stat_data(self):
        if self.end_date is None:
            self.end_date = self.start_date
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

    def add_missing_date(self, d_range, a):
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