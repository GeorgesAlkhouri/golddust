def get_data(lon=12.375089, lat=51.340349, distance=10000):
    """
    Prepare and download particulate matter data.
    """
    import ibmcloudsql
    from urllib import parse

    from api import IBM_CLOUD_CONFIG

    api_key = IBM_CLOUD_CONFIG['API_KEY']
    instancecrn_encoded = IBM_CLOUD_CONFIG['CRN']
    target_url = IBM_CLOUD_CONFIG['URL']

    # Decode the crn
    instancecrn = parse.unquote(instancecrn_encoded)

    sql_client = ibmcloudsql.SQLQuery(api_key, instancecrn, target_url)
    sql_client.logon()
    sql_client.sql_ui_link()

    st_poin_format = "ST_Point(%s, %s)" % (lon, lat)

    sql = """
    WITH prefiltered AS
         (SELECT s.sensor_id sensor_id, s.lon lon, s.lat lat, MAX(s.lon) max_lon, MAX(s.lat) max_lat , s.location location,
         min(timestamp) as start, max(timestamp) as end
          FROM cos://us-geo/sql/oklabdata/parquet/sds011/2018/07 STORED AS PARQUET s
          WHERE isnotnull(s.lon) AND isnotnull(s.lon)GROUP BY s.sensor_id, s.lon, s.lat, s.location
         )
    SELECT sensor_id, ST_Distance(ST_Point(max_lon, max_lat), %s) AS distance , location as location, max_lon as lon, max_lat as lat,
    start, end
    FROM prefiltered
    WHERE ST_Distance(ST_Point(max_lon, max_lat), %s) <= %s
    ORDER BY distance asc""" % (st_poin_format, st_poin_format, distance)

    sds011_sensors = sql_client.run_sql(sql)

    sql = """
    WITH prefiltered AS
         (SELECT s.sensor_id sensor_id, s.lon lon, s.lat lat, MAX(s.lon) max_lon, MAX(s.lat) max_lat , s.location location
          FROM cos://us-geo/sql/oklabdata/parquet/dht22/2018/07 STORED AS PARQUET s
          WHERE isnotnull(s.lon) AND isnotnull(s.lon)GROUP BY s.sensor_id, s.lon, s.lat, s.location
         )
    SELECT sensor_id, ST_Distance(ST_Point(max_lon, max_lat), ST_Point(12.375089, 51.340349)) AS distance , location as location, max_lon as lon, max_lat as lat
    FROM prefiltered
    WHERE ST_Distance(ST_Point(max_lon, max_lat), ST_Point(12.375089, 51.340349)) <= 10000.0
    ORDER BY distance asc"""

    dht22_sensors = sql_client.run_sql(sql)

    return sds011_sensors, dht22_sensors


if __name__ == "__main__":
    import argparse

    # welcher auflösung wollen wir --> stündlich
    #a, b = get_data()
