# main.py

from config import NOAA_TOKEN, STATION_ID, START_DATE, END_DATE, SAVE_CSV, CSV_PATH, SEQUENCE_LENGTH
from data_loader import fetch_ghcn_data, preprocess_ghcn
from dataset import GHCNDailyDataset
from torch.utils.data import DataLoader
import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)     

    def forward(self, x, h0=None, c0=None):
        if h0 == None or c0 == None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[: ,-1, :])
        return out, hn, cn

def main():
    df_raw = fetch_ghcn_data(NOAA_TOKEN, STATION_ID, START_DATE, END_DATE, save_csv=SAVE_CSV, csv_path=CSV_PATH)
    df_processed = preprocess_ghcn(df_raw)

    dataset = GHCNDailyDataset(df_processed, feature="TMAX", sequence_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_x, batch_y in dataloader:
        print("Input shape:", batch_x.shape)  # (batch, 7, features)
        print("Target shape:", batch_y.shape)  # (batch, features)
        break

    # torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, 
    # bidirectional=False, proj_size=0, device=None, dtype=None)[source]

    # model = LSTM(input_dim=1, hidden_dim=100, layer_dim=1, output_dim=1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # NUMBER_OF_EPOCHS = 100  # constant value, move to top for configuration
    # h0, c0 = None, None

    # for epoch in range(NUMBER_OF_EPOCHS):
    #     model.train()
    #     optimizer.zero_grad()

    #     print(f"Epoch count: ",epoch + 1)

if __name__ == "__main__":
    main()
