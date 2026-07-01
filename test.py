import pandas as pd
print(pd.read_csv("cog_lab/S1/HCI/D3_S1_mouse.csv")["type"].unique())