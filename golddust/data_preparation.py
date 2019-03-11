"""
Script for crawling and downloading sensor data from http://luftdaten.info/.
Configurable parameters are sensor type, sensor id, location_id and a optional start
date to download sensor data.
"""

import os
import multiprocessing
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import chain
from functools import reduce
import argparse
from datetime import datetime
import requests
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
from tqdm import tqdm
import d6tstack

def returning(data, key, structure):
    # new_data = copy.deepcopy(structure)
    structure[key] = data
    return structure


def parse_luftdaten_website(data):
    url = data['url']
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml', parse_only=SoupStrainer('a'))
    links = soup.findAll('a')
    return returning(links, 'links', data)


def valid_root_folder_link(link):
    try:
        _ = datetime.strptime(link, '%Y-%m-%d')
    except ValueError:
        return False
    return True


def valid_start_date_in_link(link, start_date):
    if not start_date:
        return True
    link_date = datetime.strptime(link, '%Y-%m-%d')
    return link_date >= start_date


def extract_root_folder_links(data):
    links = data['links']
    start_date = data['start_date']
    url = data['url']

    folder_links = [link['href'][0:-1] for link in links]
    folder_links = [link
                    for link in folder_links if valid_root_folder_link(link)]
    folder_links = [
        link for link in folder_links if valid_start_date_in_link(link, start_date)]
    folder_links = [url + '/' + link for link in folder_links]

    return returning(folder_links, 'folder_links', data)


def valid_sensor_in_file_link(file_link, valid_sensors):
    contains_sensor = any([sensor in file_link for sensor in valid_sensors])
    return contains_sensor


def _extract_file_links(folder_link, valid_sensors):

    soup = BeautifulSoup(requests.get(folder_link).content, 'lxml', parse_only=SoupStrainer('a'))
    links = soup.findAll('a')
    links = [link['href'] for link in links]
    return [folder_link + '/' +link for link in links if '.csv' in link and valid_sensor_in_file_link(link, valid_sensors)]


def extract_file_links(data):
    folder_links = data['folder_links']
    valid_sensors = data['valid_sensors']

    thread_pool_size = multiprocessing.cpu_count()
    with Pool(thread_pool_size) as pool:

        extract_iter = pool.uimap(lambda link: _extract_file_links(
            link, valid_sensors), folder_links)
        file_links = list(tqdm(extract_iter, total=len(folder_links)))
        file_links = list(chain.from_iterable(file_links))

    return returning(file_links, 'file_links', data)


def _contains_location(df, locations):

    if 'location' in df.columns and df.location.values.size > 0:
         ###what todo with multiple locations for same sensor?
        return str(df.location.values[0]) in locations
    return False

def _fetch_file_links(file_link, locations, save_dir):
    try:
        df = pd.read_csv(file_link, sep=';')
        if _contains_location(df, locations):
            path = save_dir + os.path.basename(file_link)
            df.to_csv(path, index=False)
            return path, None

        return None, None
    except Exception as e:
        # print(e)
        return None, file_link

def fetch_file_links(data):

    file_links = data['file_links']
    locations = data['locations']
    save_dir = data['save_dir']

    thread_pool_size = multiprocessing.cpu_count()
    with Pool(thread_pool_size) as pool:
        fetch_iter = pool.uimap(lambda link: _fetch_file_links(link, locations, save_dir), file_links)
        results = list(tqdm(fetch_iter, total=len(file_links)))
        file_paths, error_links = zip(*results)
        file_paths =  [l for l in file_paths if l is not None]
        error_links =  [l for l in error_links if l is not None]

    print('Links with valid location:', len(file_paths))
    return returning(file_paths, 'file_paths', data)

def _sort_in(d, file_path):
    if 'sds011' in file_path:
        d['sds011'].append(file_path)
    elif 'dht22' in file_path:
        d['dht22'].append(file_path)

    return d

def sort_sensor_types(data):
    file_paths_for_sensor = {'sds011': [], 'dht22': []}
    file_paths = data['file_paths']

    # Divide links into sensor types
    file_paths_for_sensor = reduce(_sort_in, file_paths, file_paths_for_sensor)

    return returning(file_paths_for_sensor, 'file_paths_for_sensor', data)

def postprocess_sensor_data(data):
    valid_sensors = data['valid_sensors']
    file_paths_for_sensor = data['file_paths_for_sensor']
    save_dir = data['save_dir']

    processed_file_paths_for_sensor = {}
    prefix = 'processed_'
    for sensor_typ in valid_sensors:
        if not file_paths_for_sensor[sensor_typ]:
            continue

        checker = d6tstack.combine_csv.CombinerCSV(
            file_paths_for_sensor[sensor_typ], sep=',')
        checker.to_csv_align(output_dir=save_dir, output_prefix=prefix)
        processed_file_paths_for_sensor[sensor_typ] = [prefix + path for path in file_paths_for_sensor[sensor_typ]]

    return returning(processed_file_paths_for_sensor, 'processed_file_paths_for_sensor', data)

def clean_up(data):
    file_paths = data['file_paths']
    for path in file_paths:
        os.remove(path)
    return data

def main(args):
    data = {}
    data['url'] = 'https://archive.luftdaten.info'
    data['locations'] = args.location
    data['save_dir'] = args.output
    data['start_date'] = args.startdate
    data['valid_sensors'] = ['sds011', 'dht22']

    parse_luftdaten_website(data)
    extract_root_folder_links(data)

    print('Extracting file links for locations:', data['locations'], 'and sensor types:', data['valid_sensors'], '...')

    extract_file_links(data)

    print('Filter locations for', len(data['file_links']), 'links and downloading...')

    fetch_file_links(data)
    sort_sensor_types(data)

    print('Postprocess data...')

    postprocess_sensor_data(data)
    clean_up(data)

# sensor_ids=['20826', '12603', '17231', '7201', '6179'],
# location_ids=['10574', '6367', '8732', '3642', '3123']
def _valid_date(parameter):
    try:
        return datetime.strptime(parameter, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'. Valid format: YYYY-MM-DD".format(parameter)
        raise argparse.ArgumentTypeError(msg)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Crawl data from luftdaten.info')

    parser.add_argument('-loc', '--location', nargs='+',help='<Required> Location IDs, separated by whitespaces.', required=True)
    parser.add_argument("-o", "--output", help='<Required> Output dir.', required=True, type=str)
    parser.add_argument("-s", "--startdate", help='<Optional> Crawl start date - format YYYY-MM-DD.', required=False, type=_valid_date)

    args = parser.parse_args()
    main(args)
