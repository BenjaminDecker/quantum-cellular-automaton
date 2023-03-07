import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from parameters import Parser

args = Parser.instance()

colormap = "inferno"
colormap2 = "viridis"


def plot(path, continuous_heatmaps: list[(np.ndarray, str)] = None,
         discrete_heatmaps: list[(np.ndarray, str)] = None) -> None:
    _, file_extension = os.path.splitext(path)
    heatmaps = continuous_heatmaps + discrete_heatmaps
    if file_extension == ".html":
        fig = make_subplots(rows=len(heatmaps))

        for index, (heatmap, label) in enumerate(heatmaps):
            discrete = index >= len(continuous_heatmaps)
            fig.add_trace(
                go.Heatmap(z=heatmap.T, coloraxis="coloraxis2" if discrete else "coloraxis"),
                index + 1,
                1
            )
            fig.update_yaxes(
                scaleanchor="x" + str(index + 1),
                row=(index + 1)
            )
        flattened_discrete_values = [item for sublist in heatmaps[len(continuous_heatmaps):] for subsublist in
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
        width, height = plt.rcParams.get("figure.figsize")
        # single_height = 2.5
        # width = single_height * (args.plot_steps/args.rules.ncells)
        fig, axs = plt.subplots(
            len(heatmaps),
            sharex='all',
            figsize=(width * 2, (height / 8) * len(heatmaps) * len(heatmaps[0]))
        )
        padding = 0.25 / width
        reference_aspect_ratio = (len(heatmaps[0][0]) / len(heatmaps[0][0][0]))
        for index, (heatmap, label) in enumerate(heatmaps):
            discrete = index >= len(continuous_heatmaps)
            if index >= len(continuous_heatmaps):
                num_ticks = int(np.max(heatmap) + 1)
            pcm = axs[index].pcolormesh(
                heatmap.T,
                cmap=plt.cm.get_cmap(colormap, num_ticks) if discrete else colormap,
                vmin=np.min(heatmap) if discrete else 0.,
                vmax=np.max(heatmap) if discrete else 1.
            )
            axs[index].set_ylabel(label.strip())

            if discrete:
                cbar = fig.colorbar(pcm, ax=axs[index], aspect=9, ticks=range(0, int(np.max(heatmap)) + 1), pad=padding)
                cbar.ax.locator_params(nbins=min(num_ticks, 5))
                pcm.set_clim(-0.5, (np.max(heatmap) + 0.5))
            axs[index].set_aspect((len(heatmap) / len(heatmap[0])) / reference_aspect_ratio)
            if index == len(continuous_heatmaps) - 1:  # last continuous plot
                ax = axs[:] if len(discrete_heatmaps) == 0 else axs[:-len(discrete_heatmaps)]
                fig.colorbar(pcm, ax=ax, pad=padding)

        plt.savefig(path, bbox_inches="tight")
