# data_loader.py

import requests
import pandas as pd
import time

def fetch_ghcn_data(token, station_id, start_date, end_date, save_csv=False, csv_path="ghcn_data.csv"):
    headers = {'token': token}
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    datasetid = "GHCND"
    limit = 1000
    offset = 1
    results = []

    while True:
        url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        params = {
            'datasetid': datasetid,
            'stationid': station_id,
            'startdate': start_date,
            'enddate': end_date,
            'limit': limit,
            'offset': offset,
            'units': 'standard'
        }

        print(f"Sending request to NOAA with:\n  Station: {station_id}\n Dates: {start_date} to {end_date}")
        print(f"URL: {url}")
        print(f"Params: {params}")

        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            print("Error:", response.json())
            break

        print("Status code:", response.status_code)
        if response.status_code != 200:
            print("Failed to fetch data:", response.text)
            return pd.DataFrame()
        
        data = response.json()
        print("Raw NOAA data keys:", data.keys())
        print("Number of results:", len(data.get("results", [])))

        data = response.json().get('results', [])
        if not data:
            break

        results.extend(data)
        offset += limit
        time.sleep(1)  # avoid rate limiting

    df = pd.DataFrame(results)
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")

    if df.empty:
        print("Warning: Empty DataFrame returned from NOAA. Check parameters or API quota.")
        return df
    
    expected_keys = {"date", "datatype", "value"}
    if not expected_keys.issubset(df.columns):
        print("Unexpected keys in API response")
        print("Available columns:", df.columns)
        print("Example row:", df.iloc[0] if not df.empty else "No rows")
        return df
    
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")

    return df


def preprocess_ghcn(df):
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    # convert 'date' column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # remove quality flags, station names, etc.
    df = df[["date", "datatype", "value"]]
    
    
    return df