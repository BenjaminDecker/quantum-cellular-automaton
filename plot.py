import os

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from parameters import Parser

args = Parser.instance()

colormap = "inferno"


def plot(heatmaps, path):
    _, file_extension = os.path.splitext(path)
    if file_extension == ".html":
        fig = make_subplots(rows=len(heatmaps))

        for index, heatmap in enumerate(heatmaps):
            fig.add_trace(
                go.Heatmap(z=heatmap.T, coloraxis="coloraxis"),
                index + 1,
                1
            )
            fig.update_yaxes(
                scaleanchor="x" + str(index + 1),
                row=(index + 1)
            )

        fig.update_layout(
            coloraxis={"colorscale": colormap, "cmax": 1.0, "cmin": 0.0}
        )
        fig.write_html(path)

    else:
        _, height = plt.rcParams.get("figure.figsize")
        ratio = args.plot_steps / (len(heatmaps) * args.rules.ncells)
        ratio = max(ratio, 1.)
        fig, axs = plt.subplots(
            len(heatmaps),
            sharex=True,
            sharey=True,
            figsize=(height * ratio, height)
        )

        for index, heatmap in enumerate(heatmaps):
            pcm = axs[index].pcolormesh(
                heatmap.T,
                cmap=colormap,
                vmin=0.,
                vmax=1.
            )
            if args.plot_steps < 50:
                axs[index].set_aspect("equal")

        fig.colorbar(pcm, ax=axs[:])

        # I use subplots_adjust to move the colorbar closer to the heatmaps
        # I don't know why, but .77 and .78 seem to give good results for their respective plot_step-ranges
        if args.rules.ncells < 15:
            if args.plot_steps <= 200:
                plt.subplots_adjust(right=.77)
            else:
                plt.subplots_adjust(right=.78)

        plt.savefig(path, bbox_inches="tight")
