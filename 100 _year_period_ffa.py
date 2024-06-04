import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearson3

# Data from the user
data = {
    'Discharge': [3105, 3092, 3057, 3000, 2953, 2915, 2798, 2740, 2731, 2717, 2575, 2565, 2552, 2529, 2495, 2464, 2456, 2449, 2444, 2432, 2385, 2345, 2342, 2333, 2313, 2271, 2245, 2240, 2236, 2235, 2182, 2177, 2081, 2056, 2014, 2010],
    'rank': list(range(1, 37)),
    'p': [0.027027027, 0.054054054, 0.081081081, 0.108108108, 0.135135135, 0.162162162, 0.189189189, 0.216216216, 0.243243243, 0.27027027, 0.297297297, 0.324324324, 0.351351351, 0.378378378, 0.405405405, 0.432432432, 0.459459459, 0.486486486, 0.513513514, 0.540540541, 0.567567568, 0.594594595, 0.621621622, 0.648648649, 0.675675676, 0.702702703, 0.72972973, 0.756756757, 0.783783784, 0.810810811, 0.837837838, 0.864864865, 0.891891892, 0.918918919, 0.945945946, 0.972972973]
}

df = pd.DataFrame(data)

# Log-Pearson Type III distribution fitting
log_discharge = np.log(df['Discharge'])
mean_log_discharge = log_discharge.mean()
std_log_discharge = log_discharge.std()
skew_log_discharge = ((log_discharge - mean_log_discharge)**3).mean() / std_log_discharge**3

# Generate the fitted distribution
rv = pearson3(skew_log_discharge, loc=mean_log_discharge, scale=std_log_discharge)

# Calculate return period for 100 years
return_period = 100
probability = 1 - 1 / return_period
log_Q100 = rv.ppf(probability)
Q100 = np.exp(log_Q100)

# Plotting with probability on the x-axis and discharge on the y-axis
plt.figure(figsize=(10, 6))

# Empirical CDF
sorted_discharge = np.sort(df['Discharge'])
empirical_cdf = np.arange(1, len(sorted_discharge) + 1) / (len(sorted_discharge) + 1)
plt.plot(empirical_cdf, sorted_discharge, marker='o', linestyle='none', label='Empirical CDF')

# Fitted Log-Pearson Type III CDF
x = np.linspace(min(log_discharge), max(log_discharge), 100)
y = rv.cdf(x)
plt.plot(y, np.exp(x), label='Log-Pearson Type III CDF')

# 100-year Return Period Discharge
plt.axhline(Q100, color='r', linestyle='--', label=f'100-year Return Period Discharge: {Q100:.2f}')

plt.ylabel('Discharge')
plt.xlabel('Cumulative Probability')
plt.title('Flood Frequency Analysis using Log-Pearson Type III Distribution')
plt.legend()
plt.grid(True)
plt.show()
