# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:48:42 2016

@author: hliu
"""

import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
from DateEvent import DateEvent, EVENT_TYPES, STD_NAMES, TIME_FORMAT


def load_json_file(jf_name):
    json_str = open(jf_name, 'r').read()
    json_str = json_str.decode('utf-8')
    event_types = json.loads(json_str)
    return event_types

#EVENT_TYPES = load_json_file('event_types.json')
#STD_NAMES = load_json_file('standard_names.json')
TIME_FORMAT = '%m %d %Y %H:%M'


def to_datetime(date_str):
    y, m, d = date_str.split('/')
    date_str = '%s %s %s 00:00' % (m, d, y)
    return datetime.strptime(date_str, TIME_FORMAT)


def to_json(obj_name, f_name):
    json_str = json.dumps(obj_name, ensure_ascii=False, indent=4)
    open(f_name, 'w').write(json_str.encode('utf-8'))


def get_raw_data(raw_name):
    d_p = re.compile(u'\d{4}/\d{1,2}/\d{1,2}')
    rm_blank = lambda x: '\n'.join([s for s in x.split('\n') if s])
    raw_data = open(raw_name, 'r').read().decode('utf-8')
    texts = d_p.split(raw_data)
    dates = d_p.findall(raw_data)
    clean_texts = [rm_blank(x) for x in texts if x]
    return dates, clean_texts


def extract_info(dates, clean_texts):
    all_events = []
    for date, text in zip(dates, clean_texts):
        all_events.append(DateEvent(to_datetime(date), text))
    return all_events


def get_existing_data(f_name):
    details = pd.read_excel(f_name, 'detail')
    stats = pd.read_excel(f_name, 'stat')
    return details, stats


def get_sleeping_time(all_dfs):
    def get_desired_time(date, time_type):
        def get_time():
            print 'date is ', date.date()
            desired_time = raw_input(u'请按照24小时输入%s时间，例如23:40\n' % time_type)
            if desired_time:
                time_str = '%d %d %d %s' % (date.month,
                                            date.day,
                                            date.year,
                                            desired_time)
                desired_time = datetime.strptime(time_str, TIME_FORMAT)
                return desired_time
            else:
                return None
        if date not in all_dfs.index:
            desired_time = get_time()
            return desired_time
        else:
            df = all_dfs.loc[date, :]
            try:
                desired_time = df.start[df.name == time_type][0]
            except IndexError:
                desired_time = get_time()
            return desired_time

    had_index = all_dfs.index[all_dfs.name == u'睡眠时长']
    dfs_no_sleeping = all_dfs.drop(had_index)
    dates = dfs_no_sleeping.index.unique()
    for date in dates:
        getup_time = get_desired_time(date, u'醒来')
        last_day_date = date + timedelta(days=-1)
        sleep_time = get_desired_time(last_day_date, u'睡觉')
        add = pd.DataFrame([[u'睡眠时长',
                             sleep_time,
                             getup_time,
                             u'睡眠',
                             u'睡眠时长']],
                           columns=['name',
                                    'start',
                                    'end',
                                    'type',
                                    'raw'],
                           index=[date])
        all_dfs = all_dfs.append(add)
    all_dfs = all_dfs.sort_index()
    return all_dfs


def merge_stat(old_details, old_stats, currt_events):
    all_types = EVENT_TYPES.keys()
    all_types.extend([u'day_active', u'unknown_type'])
    currt_raws = [e.raw_data for e in currt_events]
    currt_raws = pd.DataFrame({'raw': currt_raws})
    currt_details = pd.concat([e.events_df for e in currt_events])
    all_details = pd.concat([old_details, currt_details])
    all_details = get_sleeping_time(all_details)
    currt_stats = stat_type_time_usage(all_details.loc[currt_details.index.unique(), :])
    currt_stats = currt_stats.reindex(columns=all_types)
    currt_raws.index = currt_stats.index
    currt_stats = pd.concat([currt_raws, currt_stats], axis=1)
    old_stats = old_stats.reindex(columns=['raw']+all_types)
    all_stats = pd.concat([old_stats, currt_stats])
    all_stats = all_stats.sort_index()
    return all_details, all_stats


def save_data(all_details, all_stats):
    os.system('cp sum.xlsx sum_bk.xlsx')
    writer = pd.ExcelWriter('sum.xlsx')
    all_details.to_excel(writer, 'detail')
    all_stats.to_excel(writer, 'stat')
    writer.save()
    to_json(STD_NAMES, 'standard_names.json')
    to_json(EVENT_TYPES, 'event_types.json')


def stat_type_time_usage(dfs):
    td_2hrs = lambda x: float('%.2f' % (x.seconds/3600.))
    dates = dfs.index.unique()
    all_types = EVENT_TYPES.keys()
    all_time_usages = []
    for date in dates:
        time_usage = {}
        currt_df = dfs.loc[date, :]
        groups = currt_df.groupby('type')
        known_time_usage = 0
        for type in groups.groups.keys():
            type_df = groups.get_group(type)
            type_df = type_df.dropna()
            if len(type_df) == 0:
                time_usage['type'] = np.nan
            else:
                time = type_df.end - type_df.start
                time = td_2hrs(time.sum())
                time_usage[type] = time
            if type != u'睡眠':
                known_time_usage += time
        day_active = currt_df.start[currt_df.name==u'睡觉'] \
                     - currt_df.start[currt_df.name==u'下床']
        time_usage['day_active'] = td_2hrs(day_active[0])
        time_usage['unknown_type'] = time_usage['day_active'] - known_time_usage
        all_time_usages.append(time_usage)
    all_types.extend(['day_active', 'unknown_type'])
    time_dfs = pd.DataFrame(dict((col, [x.get(col, np.nan) for x in all_time_usages]) for col in all_types),
                            columns=all_types,
                            index=dates)
    return time_dfs


def main():
    dates, clean_texts = get_raw_data('raw_data.dat')
    old_details, old_stats = get_existing_data('sum.xlsx')
    currt_events = extract_info(dates, clean_texts)
    all_details, all_stats = merge_stat(old_details, old_stats, currt_events)
    save_data(all_details, all_stats)