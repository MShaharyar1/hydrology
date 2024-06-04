import numpy as np
import pandas as pd
from scipy.stats import pearson3
import matplotlib.pyplot as plt

# Data from the user
data = {
    'Discharge': [3105, 3092, 3057, 3000, 2953, 2915, 2798, 2740, 2731, 2717, 2575, 2565, 2552, 2529, 2495, 2464, 2456, 2449, 2444, 2432, 2385, 2345, 2342, 2333, 2313, 2271, 2245, 2240, 2236, 2235, 2182, 2177, 2081, 2056, 2014, 2010],
    'rank': list(range(1, 37)),
    'p': [0.027027027, 0.054054054, 0.081081081, 0.108108108, 0.135135135, 0.162162162, 0.189189189, 0.216216216, 0.243243243, 0.27027027, 0.297297297, 0.324324324, 0.351351351, 0.378378378, 0.405405405, 0.432432432, 0.459459459, 0.486486486, 0.513513514, 0.540540541, 0.567567568, 0.594594595, 0.621621622, 0.648648649, 0.675675676, 0.702702703, 0.72972973, 0.756756757, 0.783783784, 0.810810811, 0.837837838, 0.864864865, 0.891891892, 0.918918919, 0.945945946, 0.972972973],
    'return_period': [37, 18.5, 12.33333333, 9.25, 7.4, 6.166666667, 5.285714286, 4.625, 4.111111111, 3.7, 3.363636364, 3.083333333, 2.846153846, 2.642857143, 2.466666667, 2.3125, 2.176470588, 2.055555556, 1.947368421, 1.85, 1.761904762, 1.681818182, 1.608695652, 1.541666667, 1.48, 1.423076923, 1.37037037, 1.321428571, 1.275862069, 1.233333333, 1.193548387, 1.15625, 1.121212121, 1.088235294, 1.057142857, 1.027777778]
}

df = pd.DataFrame(data)

# Log-Pearson Type III distribution fitting
log_discharge = np.log(df['Discharge'])
mean_log_discharge = log_discharge.mean()
std_log_discharge = log_discharge.std()
skew_log_discharge = ((log_discharge - mean_log_discharge)**3).mean() / std_log_discharge**3

# Generate the fitted distribution
rv = pearson3(skew_log_discharge, loc=mean_log_discharge, scale=std_log_discharge)

# Define a range of return periods
return_periods = np.logspace(0.1, 2.7, num=100)  # Range from about 1.26 to 500 years
probabilities = 1 - 1 / return_periods

# Calculate the corresponding discharges
log_Q = rv.ppf(probabilities)
Q = np.exp(log_Q)

# Plotting Flow vs Return Period with fitted Log-Pearson Type III curve
plt.figure(figsize=(10, 6))
plt.plot(df['return_period'], df['Discharge'], marker='o', linestyle='none', color='b', label='Observed Data')
plt.plot(return_periods, Q, linestyle='-', color='r', label='Fitted Log-Pearson Type III')

plt.xlabel('Return Period (years)')
plt.ylabel('Discharge (m³/s)')
plt.title('Flow vs Return Period using Log-Pearson Type III Distribution')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
