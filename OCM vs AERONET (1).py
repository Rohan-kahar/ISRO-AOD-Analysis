#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress

# === EDIT THIS PATH ==========================================================
csv_path = r"C:\Users\admin\rohan\Kanpur.csv"      # <- change to your CSV
# ============================================================================

# 1. Load & parse the data ----------------------------------------------------
df = pd.read_csv(csv_path, parse_dates=['date'])
df = df.sort_values('date')          # ensure chronological order

# Global font sizes for all figs
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# 2. Grouped bar plot ---------------------------------------------------------
plt.style.use('default')
fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
bar_width = 0.4
x = range(len(df))

ax1.bar([i - bar_width/2 for i in x], df['AOD_M'], width=bar_width, label='OCM (AOD_M)')
ax1.bar([i + bar_width/2 for i in x], df['AOD_N'], width=bar_width, label='AERONET (AOD_N)')

ax1.set_title('Kanpur AOD Comparison: OCM vs. AERONET')
ax1.set_xlabel('Date')
ax1.set_ylabel('AOD')
ax1.set_xticks(x)
ax1.set_xticklabels(df['date'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
ax1.yaxis.set_major_locator(MaxNLocator(integer=False))
ax1.legend(loc='best', frameon=False)

fig1.tight_layout()
fig1.savefig('bar_AOD_OCM_vs_AERONET.png', bbox_inches='tight')

plt.show()

