import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from parameters import Parser

args = Parser.instance()

colormap = "inferno"
colormap2 = "viridis"


def plot(path, probability_heatmaps: list[(np.ndarray, str)] = None,
         discrete_heatmaps: list[(np.ndarray, str)] = None) -> None:
    _, file_extension = os.path.splitext(path)
    heatmaps = probability_heatmaps + discrete_heatmaps
    if file_extension == ".html":
        fig = make_subplots(rows=len(heatmaps))

        for index, (heatmap, label) in enumerate(heatmaps):
            discrete = index >= len(probability_heatmaps)
            fig.add_trace(
                go.Heatmap(z=heatmap.T, coloraxis="coloraxis2" if discrete else "coloraxis"),
                index + 1,
                1
            )
            fig.update_yaxes(
                scaleanchor="x" + str(index + 1),
                row=(index + 1)
            )
        flattened_discrete_values = [item for sublist in heatmaps[len(probability_heatmaps):] for subsublist in
                                     sublist[0] for item in subsublist]
        if len(flattened_discrete_values) > 0:
            fig.update_layout(
                coloraxis2={
                    "colorscale": colormap2,
                    "cmax": max(1.0, max(flattened_discrete_values)),
                    "cmin": min(flattened_discrete_values),
                    "colorbar_xpad": 100
                }
            )
        fig.update_layout(
            coloraxis={"colorscale": colormap, "cmax": 1., "cmin": 0.0}
        )
        fig.write_html(path)

    else:
        fig, axs = plt.subplots(
            len(heatmaps),
            sharex='all'
        )
        reference_aspect_ratio = (len(heatmaps[0][0]) / len(heatmaps[0][0][0]))
        for index, (heatmap, label) in enumerate(heatmaps):
            discrete = index >= len(probability_heatmaps)
            if index >= len(probability_heatmaps):
                num_ticks = int(np.max(heatmap) - np.min(heatmap) + 1)
            pcm = axs[index].pcolormesh(
                heatmap.T,
                cmap=plt.cm.get_cmap(colormap, num_ticks) if discrete else colormap,
                vmin=np.min(heatmap) if discrete else 0.,
                vmax=np.max(heatmap) if discrete else 1.
            )
            if discrete:
                fig.colorbar(pcm, ax=axs[index], aspect=9, label=label,
                             ticks=range(int(np.min(heatmap)), int(np.max(heatmap)) + 1))
                pcm.set_clim(np.min(heatmap) - 0.5, np.max(heatmap) + 0.5)
            axs[index].set_aspect((len(heatmap) / len(heatmap[0])) / reference_aspect_ratio)
            if index == len(probability_heatmaps) - 1:  # last probability plot
                ax = axs[:] if len(discrete_heatmaps) == 0 else axs[:-len(discrete_heatmaps)]
                fig.colorbar(pcm, ax=ax, label=label)

        plt.savefig(path, bbox_inches="tight")
