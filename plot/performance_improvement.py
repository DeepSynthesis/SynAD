import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = [
    ["ULD", 5141, "yield", "literature-based splitting", -0.20, 0.91, "12.7%"],
    ["CPA-catalyzed", 365, "ddG", "ligand-based splitting", 0.10, 0.42, "28.2%"],
    ["NiCOlit", 1804, "yield", "literature-based splitting", -0.53, 0.19, "10.4%"],
    ["Pd-CO-catalyzed", 2499, "yield", "literature-based splitting", -0.12, 0.52, "4.3%"],
    ["B-H HTE", 4599, "yield", "ligand-based splitting", 0.30, 0.91, "25.4%"],
    ["CPA HTE", 1075, "ddG", "catalyst-based splitting", 0.56, 0.70, "86.0%"],
    ["Suzuki HTE", 5760, "yield", "ligand-based splitting", 0.27, 0.45, "64.1%"],
]
df = pd.DataFrame(
    data, columns=["Dataset", "Reaction number", "Predict target", "Data partition method", "original R2", "In-SynAD R2", "Coverage"]
)

df["is_HTE"] = df["Dataset"].str.contains("HTE")
df_long = pd.melt(
    df, id_vars=["Dataset", "Predict target", "is_HTE"], value_vars=["original R2", "In-SynAD R2"], var_name="Method", value_name="R2"
)
df_long["Method"] = df_long["Method"].str.replace(" R2", "")
df_long["Category"] = df_long["Dataset"]
category_order = [
    "ULD",
    "CPA-catalyzed",
    "NiCOlit",
    "Pd-CO-catalyzed",
    "B-H HTE",
    "CPA HTE",
    "Suzuki HTE",
]
df_long["Category"] = pd.Categorical(df_long["Category"], categories=category_order, ordered=True)

color_map = {(False, "original"): "#a1c6e4", (False, "In-SynAD"): "#5b97c8", (True, "original"): "#db8186", (True, "In-SynAD"): "#c6262f"}
df_long["Color"] = df_long.apply(lambda row: color_map[(row["is_HTE"], row["Method"])], axis=1)

plt.figure(figsize=(16, 8))
ax = plt.gca()
bar_width = 0.35

for i, category in enumerate(category_order):
    sub_df = df_long[df_long["Category"] == category].sort_values("Method")
    x_pos = [i + bar_width / 2, i - bar_width / 2]
    ax.bar(x_pos, sub_df["R2"], width=bar_width, color=sub_df["Color"], edgecolor="gray")

plt.xticks(range(len(category_order)), category_order, rotation=45, ha="right", fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Dataset", fontsize=20)
plt.ylabel("R² Value", fontsize=20)
plt.axhline(y=0, color="gray", linestyle="--", linewidth=1)
plt.ylim(-0.6, 1.0)
sns.despine(top=True, right=True)

for p in ax.patches:
    height = p.get_height()
    va = "bottom" if height > 0 else "top"
    offset = 0.02 if height > 0 else -0.04
    ax.text(p.get_x() + p.get_width() / 2, height + offset, f"{height:.2f}", ha="center", va=va, fontsize=15)

plt.tight_layout()
plt.savefig("performance_improvement.png", dpi=300)
plt.show()
