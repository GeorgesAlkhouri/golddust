"""
Script for crawling and downloading sensor data from http://luftdaten.info/.
Configurable parameters are sensor type, sensor id, location_id and a optional start
date to download sensor data.
"""

import os
import multiprocessing
from multiprocessing import Pool
from datetime import datetime
from itertools import chain
from functools import reduce
import requests
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
from tqdm import tqdm
import d6tstack

sensors = ['sds011', 'dht22', 'bme280']
sensor_ids = ['20826', '6179', '12603', '17231', '7201']
location_ids = ['10574', '3123', '6367', '8732', '3642']
start_date = datetime.strptime('2018-01-01', '%Y-%m-%d')
save_dir = 'data/'

url = 'https://archive.luftdaten.info'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')


def valid_link(link):
    try:
        link_date = datetime.strptime(link['href'][0:-1], '%Y-%m-%d')
    except ValueError:
        return False
    else:
        if not start_date:
            return True
        else:
            return link_date >= start_date


def valid_sensor(link):
    contains_sensor = any([sensor in link for sensor in sensors])
    # contains_sensor_id = any([sensor_id in link for sensor_id in sensor_ids])

    # return contains_sensor and contains_sensor_id
    return contains_sensor


def get_file_links(link):
    soup = BeautifulSoup(requests.get(link).content,
                         'lxml', parse_only=SoupStrainer('a'))
    links = soup.findAll('a')
    return [link + l['href'] for l in links if '.csv' in l['href'] and valid_sensor(l['href'])]


def fetch_csv(link):
    return pd.read_csv(link, sep=';')


def contains_location(df):
    if 'location' in df.columns and len(df.location) > 0:
        return str(df.location.values[0]) in location_ids
    else:
        return False


def fetch(link):
    try:
        df = fetch_csv(link)
        res = contains_location(df)
        if res:
            path = save_dir + os.path.basename(link)
            df.to_csv(path, index=False)
            return path, None
        else:
            return None, None
    except:
        return None, link


links = [link['href'] for link in soup.findAll('a') if valid_link(link)]
links = [url + '/' + link for link in links]

print("Start crawling %s links beginning from the %s" %
      (str(len(links)), start_date if start_date else 'beginning'))

thread_pool_size = multiprocessing.cpu_count()

with Pool(thread_pool_size) as pool:

    links = list(tqdm(pool.imap_unordered(
        get_file_links, links), total=len(links)))
    links = list(chain.from_iterable(links))
    print(list(links[0:10]))
    print('Start to analyse %s links for same location ids' % str(len(links)))

    links, error_links = zip(*list(tqdm(pool.imap_unordered(
        fetch, links), total=len(links))))

    pool.close()
    pool.join()


links = [l for l in links if l is not None]


print('Links with errors')
print(error_links)

print('Post processing downloaded files')

links_for_sensor = {'bme280': [], 'sds011': [], 'dht22': []}


def sort_in(d, link):
    if 'bme280' in link:
        d['bme280'].append(link)
    elif 'sds011' in link:
        d['sds011'].append(link)
    elif 'dht22' in link:
        d['dht22'].append(link)
    else:
        d['none'].append(link)
    return d


# Divide links into sensor types
links_for_sensor = reduce(sort_in, links, links_for_sensor)

print('Link count for sensor type')
print(len(links_for_sensor['bme280']), len(links_for_sensor['sds011']), len(
    links_for_sensor['dht22']))

for sensor_typ in sensors:
    if not links_for_sensor[sensor_typ]:
        continue
    print(sensor_typ)

    c = d6tstack.combine_csv.CombinerCSV(links_for_sensor[sensor_typ], sep=';')
    c.to_csv_align(output_dir=save_dir, output_prefix='processed_')
