import time
from bs4 import BeautifulSoup, SoupStrainer
import requests
import re
from datetime import datetime
from itertools import chain
import pandas as pd
from functools import reduce
import dask.dataframe as dd
from tqdm import tqdm
import os
import multiprocessing
from multiprocessing import Pool
import d6tstack

sensors = ['sds011', 'dht22', 'bme280']
sensor_ids = ['20826', '6179', '12603', '17231', '7201']
save_dir = 'data/'


url = 'https://archive.luftdaten.info'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')


def valid_link(link):
    try:
        datetime.strptime(link['href'][0:-1], '%Y-%m-%d')
    except ValueError:
        return False
    else:
        return True


def valid_sensor(link):
    contains_sensor = any([sensor in link for sensor in sensors])
    contains_sensor_id = any([sensor_id in link for sensor_id in sensor_ids])

    return contains_sensor and contains_sensor_id


def get_file_links(link):
    soup = BeautifulSoup(requests.get(link).content,
                         'lxml', parse_only=SoupStrainer('a'))
    links = soup.findAll('a')
    return [link + l['href'] for l in links if '.csv' in l['href'] and valid_sensor(l['href'])]


def fetch_url(link):
    path = save_dir + os.path.basename(link)
    if not os.path.exists(path):
        r = requests.get(link, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
    return path


links = [link['href'] for link in soup.findAll('a') if valid_link(link)]
links = [url + '/' + link for link in links]


print("Start crawling %s links" % str(len(links)))

thread_pool_size = multiprocessing.cpu_count()

with Pool(thread_pool_size) as pool:

    links = tqdm(pool.imap_unordered(
        get_file_links, links), total=len(links))
    links = list(chain.from_iterable(links))

    print('Start downloading %s links to %s' % (str(len(links)), save_dir))

    links = tqdm(pool.imap_unordered(
        fetch_url, links), total=len(links))

    pool.close()
    pool.join()

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
