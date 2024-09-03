# Select the smallest ttc in one frameid


import pandas as pd
from io import StringIO

# Correct the OCR extracted text and format it into a csv-like structure
filepath = "D:/Trajectory/2.Merge_B/2.Rear-end_output_TTC_04.csv"

# Define the path for the new csv file
output_csv_path = r"D:/Trajectory/2.Merge_B/MB_smallest_TTC_d.csv"

# Convert the corrected text into a pandas DataFrame
data = pd.read_csv(filepath)

# target laneid
target_laneids= [7, 8, 9, 10]

# 根据 laneid 过滤数据
filtered_data = data[data['laneid'].isin(target_laneids)]

# Group by 'frameid' and get the index of the minimum 'TTC' for each group
idx_min_ttc = filtered_data.groupby('frameid')['TTC'].idxmin()

# Select the rows with the minimum 'TTC' for each 'frameid'
min_ttc_data = filtered_data.loc[idx_min_ttc]

# Reset index to clean up the DataFrame
min_ttc_data.reset_index(drop=True, inplace=True)

# Save the DataFrame with the minimum TTC data to a csv file
min_ttc_data.to_csv(output_csv_path, index=False)
