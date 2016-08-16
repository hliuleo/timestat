# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:17:29 2016

@author: hliu
"""
import re
from datetime import datetime, timedelta
import pandas as pd
import json


def load_json_file(jf_name):
    json_str = open(jf_name, 'r').read()
    json_str = json_str.decode('utf-8')
    event_types = json.loads(json_str)
    return event_types

EVENT_TYPES = load_json_file('event_types.json')
STD_NAMES = load_json_file('standard_names.json')
TIME_FORMAT = '%m %d %Y %H:%M'


class DateEvent:

    def __init__(self, date, text):
        text = re.sub(u'。', u'，', text)
        self.raw_data = text
        self.date = date
        self._events = []
        self.events_df = self.txt2DF()


    def txt2DF(self):
        point_p = re.compile(u'\d{1,2}:\d{2}')
        event_p = re.compile(u'\d{1,2}:\d{2}[^，]+，')
        extra_span_p = re.compile(u'中间[^\d]*，')
        noon_str = '%02d %d %d 12:00' % (self.date.month,
                                         self.date.day,
                                         self.date.year)
        noon = datetime.strptime(noon_str, TIME_FORMAT)

        def is_time_span(string):
            span_p = re.compile(u'\d{1,2}:\d{2}结束')
            match = span_p.search(string)
            return match != None

        def has_extra_span(string):
            match = extra_span_p.search(string)
            return match != None

        def get_event_name(string):
            event = point_p.split(string)[1][:-1]
            return event

        def get_extra_event_name(string):
            last_event = extra_span_p.search(string).group()
            last_event = last_event[2:-1]
            return last_event

        def save_events(name, start, end=None):
            event = {'raw': name, 'start': start, 'end': end}
            self._events.append(event)

        def to_date(time_str):
            datetime_str = '%02d %d %d %s' % (self.date.month,
                                              self.date.day,
                                              self.date.year,
                                              time_str)
            date_time = datetime.strptime(datetime_str, TIME_FORMAT)
            if passed_noon and (date_time < noon):
                date_time += timedelta(days=1)
            if last:
                if date_time - last > timedelta(days=0.5):
                    date_time -= timedelta(days=0.5)
            return date_time

        data = self.raw_data.split(u'\n')
        last = False
        passed_noon = False
        for i in data:
            if is_time_span(i):
                try:
                    start, end = map(to_date, point_p.findall(i))
                except ValueError:
                    print i
                last = end
                if has_extra_span(i):
                    name = get_extra_event_name(i)
                    extra_end = start
                    save_events(name,
                                extra_start,
                                extra_end)
                name = get_event_name(event_p.search(i).group())
                extra_start = end
                if not passed_noon:
                    passed_noon = end >= noon
                save_events(name,
                            start,
                            end)

            else:
                names = map(get_event_name, event_p.findall(i))
                points = map(to_date, point_p.findall(i))
                try:
                    last = points[-1]
                except:
                    print i, names
                is_single_point = len(points) == 1
                if not passed_noon:
                    passed_noon = points[-1] >= noon
                if has_extra_span(i):
                    extra_name = get_extra_event_name(i)
                    if is_single_point:
                        extra_end = points[0]
                        save_events(extra_name,
                                    extra_start,
                                    extra_end)
                    else:
                        start, end = points
                        save_events(extra_name,
                                    start,
                                    end)
                for p, n in zip(points, names):
                    save_events(n, p)
                extra_start = points[-1]

        self.get_event_type()
        self._events = sorted(self._events, key=lambda x: x['start'])
        cols = ['name', 'start', 'end', 'type', 'raw']
        events_df = dict((col, map(lambda x: x.get(col, 'TEMP'), self._events)) for col in cols)
        events_df = pd.DataFrame(self._events, columns=cols)
        events_df.index = [self.date] * len(events_df)
        return events_df

    def get_event_type(self):
        for event in self._events:
            raw_name = event['raw']
            for key in STD_NAMES.keys():
                std_names = STD_NAMES[key]
                if raw_name in std_names:
                    event['name'] = key
                    s_name = key
                    break
            else:
                print '%d/%d/%d' % (self.date.year, self.date.month, self.date.day)
                print u'可改为：%s' % (u'、'.join(STD_NAMES.keys()))
                print u'event是%s' % raw_name
                s_name = raw_input(u'请输入标准名称：').decode('utf-8')
                event['name'] = s_name
                if s_name in STD_NAMES.keys():
                    save_flag = raw_input(u'是否记住此次名称更改').decode('utf-8')
                    if save_flag in ['y', 'yes']:
                        STD_NAMES[s_name].append(raw_name)
                else:
                    save_flag = raw_input(u'是否添加此标准名称').decode('utf-8')
                    if save_flag in ['y', 'yes']:
                        STD_NAMES[s_name] = [raw_name]
            for type_name in EVENT_TYPES.keys():
                avaiable_list = EVENT_TYPES[type_name]
                if s_name in avaiable_list:
                    event['type'] = type_name
                    break
            else:
                print u'可选类型为：%s' % (u'、'.join(EVENT_TYPES.keys()))
                print u'标准名称是：%s' % s_name
                t_name = raw_input(u'请输入类型名称：').decode('utf-8')
                event['type'] = t_name
                if t_name in EVENT_TYPES.keys():
                    save_flag = raw_input(u'是否添加此标准名称').decode('utf-8')
                    if save_flag in ['y', 'yes']:
                        EVENT_TYPES[t_name].append(s_name)
                else:
                    save_flag = raw_input(u'是否添加此类型').decode('utf-8')
                    if save_flag in ['y', 'yes']:
                        EVENT_TYPES[t_name] = [s_name]