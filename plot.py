from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from parameters import args
import os

colormap = "inferno"


def plot(heatmaps, path):
    _, file_extension = os.path.splitext(path)
    if file_extension == ".html":
        fig = make_subplots(rows=len(heatmaps))

        for index, heatmap in enumerate(heatmaps):
            fig.add_trace(go.Heatmap(
                z=heatmap.T, coloraxis="coloraxis"), index + 1, 1)
            fig.update_yaxes(scaleanchor=(
                "x" + str(index + 1)), row=(index + 1))

        fig.update_layout(
            coloraxis={"colorscale": colormap, "cmax": 1.0, "cmin": 0.0})
        fig.write_html(path)

    else:
        width, height = plt.rcParams.get("figure.figsize")
        ratio = args.NUM_STEPS / (len(heatmaps) * args.NUM_CELLS)
        ratio = max(ratio, 1.)
        fig, axs = plt.subplots(
            len(heatmaps), sharex=True, sharey=True, figsize=(width * ratio, height))

        for index, heatmap in enumerate(heatmaps):
            pcm = axs[index].pcolormesh(
                heatmap.T, cmap=colormap, vmin=0., vmax=1.)
            axs[index].set_aspect("equal")

        fig.colorbar(pcm, ax=axs[:])
        plt.savefig(path, bbox_inches="tight")
