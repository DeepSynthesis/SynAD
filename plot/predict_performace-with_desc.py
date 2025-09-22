import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

text_size = 20
plt.rcParams.update(
    {
        "font.size": text_size,
        "axes.labelsize": text_size,
        "axes.titlesize": text_size,
        "xtick.labelsize": text_size,
        "ytick.labelsize": text_size,
        "font.family": "Arial",
    }
)

COLOR_PALETTE = {"SPOC": "#5b97c8", "Q-SPOC": "#464b84", "QPOC": "#c6262f"}


def plot_with_different_partition(dataset_type, return_figure=False):
    data_dict = {
        "Random": {
            "Q-SPOC": [0.6187566005736951, 0.598731127060889, 0.5944312512032501, 0.6066235874024406, 0.5851643012613724],
            "QPOC": [0.5851262766808055, 0.5242567040057897, 0.5498367273524947, 0.5497180881443899, 0.5493199129237498],
            "SPOC": [0.5819196693187714, 0.5733263909631066, 0.5609570462734623, 0.5662518037670473, 0.5719000001737926],
        },
        "Literature": {
            "Q-SPOC": [-0.4601985979963601, -0.950392550257233, -0.2456151737665888, 0.22177109479579593, -0.017931156147425575],
            "QPOC": [-0.21620573019220624, -0.7377711682190171, -0.10384884559282437, 0.21651281904537412, 0.01927075175697157],
            "SPOC": [-0.2563580596402386, -1.062705297253033, -0.2558231795869357, 0.15847918743573342, -0.09439804177043953],
        },
        "Ligand": {
            "Q-SPOC": [-0.7592591203225969, -0.24872740769972057, -0.14743396660714403, 0.05259034412934527, 0.02029509728404355],
            "QPOC": [-0.8513720084686345, -0.23566430776358627, 0.024741549800141427, 0.13376388365686587, -0.08500150164698095],
            "SPOC": [-0.8248464162373248, -0.25519258075326445, -0.08978129182029226, 0.04901887852188325, 0.10384296351008415],
        },
        "Year": {
            "Q-SPOC": [-0.8100404362422502],
            "QPOC": [-0.41024803032590906],
            "SPOC": [-0.900679254489059],
        },
    }

    data_list = []
    for partition in ["Random", "Literature", "Ligand", "Year"]:
        for method, values in data_dict[partition].items():
            for value in values:
                data_list.append({"Partition": partition, "Method": method, "Value": value})

    df = pd.DataFrame(data_list)
    grouped = df.groupby(["Partition", "Method"]).mean().reset_index()

    for partition in ["Random", "Literature", "Ligand", "Year"]:
        plt.figure(figsize=(8, 8))
        partition_data = df[df["Partition"] == partition]

        ax = sns.barplot(
            x="Method",
            y="Value",
            data=partition_data,
            hue="Method",
            palette=COLOR_PALETTE,
            edgecolor="black",
            width=0.6,
            estimator=np.mean,
            errorbar=None,
            legend=False,
        )

        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2.0,
                height * 0.95,
                f"{height:.2f}",
                ha="center",
                va="top" if height > 0 else "bottom",
                color="white",
                fontsize=text_size - 2,
                fontweight="bold",
            )

        plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
        plt.title(f"Partition: {partition}", fontsize=text_size, pad=20)
        plt.ylabel("R²", fontweight="bold")
        plt.ylim(-1.0, 0.8)

        save_path = Path(__file__).parent / f"barplot_{partition}_partition_{dataset_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    plot_with_different_partition("ULD")
