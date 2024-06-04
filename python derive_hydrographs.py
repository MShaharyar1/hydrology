import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = 'Tarbela Data for CEP (MS 303) (1).xlsx'
xls = pd.ExcelFile('Tarbela Data for CEP (MS 303) (1).xlsx')

# Load the data from the first sheet
data = pd.read_excel(xls, sheet_name='Sheet1')

# Clean the data by removing the first two rows which are headers and irrelevant
cleaned_data = data.iloc[2:]

# Rename the columns using the first valid row as header
cleaned_data.columns = cleaned_data.iloc[0]
cleaned_data = cleaned_data.drop(cleaned_data.index[0])

# Reset index for clean dataframe
cleaned_data = cleaned_data.reset_index(drop=True)

# Convert columns to appropriate data types
cleaned_data = cleaned_data.apply(pd.to_numeric, errors='ignore')

# Remove columns that have all NaN values
cleaned_data = cleaned_data.dropna(axis=1, how='all')

# Set the index to 'Month/10 Daily'
cleaned_data.rename(columns={np.nan: 'NA', 'Month/10 Daily ': 'Month/10 Daily'}, inplace=True)
cleaned_data.set_index('Month/10 Daily', inplace=True)

# Calculate the average, maximum, and minimum flow for each 10-day period
cleaned_data['Average Flow'] = cleaned_data.mean(axis=1)
cleaned_data['Maximum Flow'] = cleaned_data.max(axis=1)
cleaned_data['Minimum Flow'] = cleaned_data.min(axis=1)

# Plot the hydrographs
plt.figure(figsize=(14, 8))

plt.plot(cleaned_data.index, cleaned_data['Average Flow'], label='Average Flow')
plt.plot(cleaned_data.index, cleaned_data['Maximum Flow'], label='Maximum Flow')
plt.plot(cleaned_data.index, cleaned_data['Minimum Flow'], label='Minimum Flow')

plt.xlabel('Time (10-day periods)')
plt.ylabel('Flow (Cumecs)')
plt.title('Hydrographs for Tarbela Dam')
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
