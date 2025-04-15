# main.py

from config import NOAA_TOKEN, STATION_ID, START_DATE, END_DATE, SAVE_CSV, CSV_PATH, SEQUENCE_LENGTH
from data_loader import fetch_ghcn_data, preprocess_ghcn
from dataset import GHCNDailyDataset
from torch.utils.data import DataLoader

def main():
    df_raw = fetch_ghcn_data(NOAA_TOKEN, STATION_ID, START_DATE, END_DATE, save_csv=SAVE_CSV, csv_path=CSV_PATH)
    df_processed = preprocess_ghcn(df_raw)

    dataset = GHCNDailyDataset(df_processed, feature="TMAX", sequence_length=7)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_x, batch_y in dataloader:
        print("Input shape:", batch_x.shape)  # (batch, 7, features)
        print("Target shape:", batch_y.shape)  # (batch, features)
        break

if __name__ == "__main__":
    main()
