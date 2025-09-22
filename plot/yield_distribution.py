import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel("../data/ULD.xlsx")
yield_cols = data.columns[data.columns.str.contains("yield", case=False)]
yield_ = data[yield_cols[0]]

fig, ax = plt.subplots(figsize=(12, 8))
sns.histplot(x=yield_, binwidth=10, bins=range(0, 101, 10), edgecolor="black", facecolor="#75ABD7")
ax.set_xlim(0, 100)
ax.set_xlabel("Yield/%", fontsize=20)
ax.set_ylabel("Frequency", fontsize=20)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
sns.set_style("whitegrid", {"grid.color": ".9"})
plt.tight_layout()
plt.savefig("yield_distribution.png", dpi=300, bbox_inches="tight")
