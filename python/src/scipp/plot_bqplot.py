# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
# @file
# @author Neil Vaytet

import numpy as np
from bqplot import pyplot as plt
import bqplot as bq
from .tools import edges_to_centers, axis_label, parse_colorbar, \
                   process_dimensions
from ipywidgets import VBox, HBox
from IPython.display import display



def plot_bqplot(input_data, ndim=0, name=None, config=None, **kwargs):
    """
    Function to automatically dispatch the input dataset to the appropriate
    plotting function depending on its dimensions
    """

    if ndim == 1:
        return __plot_1d(input_data, **kwargs)
    elif ndim == 2:
        return __plot_2d(input_data, name=name, config=config, **kwargs)
    elif ndim > 2:
        raise RuntimeError("Plotting for 3 and more dimensions in bqplot "
                           "is not yet implemented. Try using the plotly "
                           "backend instead by setting "
                           "scipp.plot_config.backend = 'plotly'.")
    else:
        raise RuntimeError("Invalid number of dimensions for "
                           "plotting: {}".format(ndim))
    return


def __plot_1d(input_data, logx=False, logy=False, logxy=False, axes=None,
              color=None, filename=None, title=""):
    """
    Plot a 1D spectrum.

    Input is a dictionary containing a list of DataProxy.
    If the coordinate of the x-axis contains bin edges, then a bar plot is
    made.

    TODO: find a more general way of handling arguments to be sent to plotly,
    probably via a dictionay of arguments
    """

    print(type(input_data))

    # fig = bq.Figure(title=title)
    x_sc = bq.LinearScale()
    y_sc = bq.LinearScale()
    ax_x = bq.Axis(scale=x_sc)
    ax_y = bq.Axis(scale=y_sc, orientation='vertical')
    # ax = fig.add_subplot(111)
    # axes_options = {}

    color_count = 0

    var = input_data[next(iter(input_data))]
    xcoord = var.coords[var.dims[0]]
    x = xcoord.values
    ax_x.label = axis_label(xcoord)
    xc = edges_to_centers(x)[0]

    lines = {"y": [], "labels": [], "colors": []}
    bars = {"y": [], "labels": [], "colors": []}

    # line_list = []
    # line_labs = []
    # line_cols = []
    # bar_list = []
    # bar_labs = []
    # bar_cols = []
    marks = []
    # print(color)

    for name, var in input_data.items():
        # xcoord = var.coords[var.dims[0]]
        # x = xcoord.values
        # xlab = axis_label(xcoord)
        y = var.values
        ylab = axis_label(var=var, name=name)
        # axes_options = {'x': {'label': xlab}}#, 'y': {'label': ylab}}
        # ax_x.label = xlab
        # if var.variances is not None:
        #     err = np.sqrt(var.variances)

        # Check for bin edges
        if x.shape[0] == y.shape[0] + 1:
            # xe = x.copy()
            # ye = np.concatenate(([0], y))
            # x, w = edges_to_centers(x)
            # if var.variances is not None:
            #     yerr = np.sqrt(var.variances)
            # else:
            #     yerr = None
            # plt.bar(x, y, padding=0.0, labels=[ylab], opacities=[0.6]*len(x),
            #        colors=[color[color_count]], stroke=color[color_count])
            # bars["y"].append(y)
            # bars["labels"].append(ylab)
            # bars["colors"].append(color[color_count])
            marks.append(bq.Bars(x=xc, y=y, scales={'x': x_sc, 'y': y_sc},
                                 padding=0.0, labels=[ylab],
                                 # opacities=[0.6]*len(xc),
                                 colors=[color[color_count]],
                                 stroke=color[color_count]))
            xerr = xc
        else:
            # # Include variance if present
            # if var.variances is not None:
            #     err = np.sqrt(var.variances)
            #     # y_var = []
            #     # for i in range(len(y)):
            #     #     err = np.sqrt(var.variances)
            #     #     marks.append(bq.Lines(x=[x[i]]*2, y=[y[i]-err, y[i]+err],
            #     #                           scales={'x': x_sc, 'y': y_sc},
            #     #                           colors=[color[color_count]]))
            #     y_var = [ [y[i], y[i] + err[i], y[i] - err[i], y[i]] for i in range(len(y))]
            #     marks.append(bq.OHLC(x=x, y=y_var, marker='candle', scales={'x': x_sc, 'y': y_sc},
            #         format='ohlc',
            # stroke=color[color_count], display_legend=False))
            #     # ax.errorbar(x, y, yerr=np.sqrt(var.variances),
            #     #             label=ylab, color=color[color_count],
            #     #             ecolor=color[color_count])
            # else:
            #     # ax.plot(x, y, label=ylab, color=color[color_count])
            # line = plt.plot(x, y, axes_options=axes_options, display_legend=True, labels=[ylab], colors=[color[color_count]])
            # line = bq.Lines(x=x, y=y,
            #  scales={'x': x_sc, 'y': y_sc})
            # lines["y"].append(y)
            # lines["labels"].append(ylab)
            # lines["colors"].append(color[color_count])
            marks.append(bq.Lines(x=x, y=y, scales={'x': x_sc, 'y': y_sc},
                                  labels=[ylab], colors=[color[color_count]]))
            xerr = x

        if var.variances is not None:
            err = np.sqrt(var.variances)
            # y_var = []
            # for i in range(len(y)):
            #     err = np.sqrt(var.variances)
            #     marks.append(bq.Lines(x=[x[i]]*2, y=[y[i]-err, y[i]+err],
            #                           scales={'x': x_sc, 'y': y_sc},
            #                           colors=[color[color_count]]))
            y_var = [ [y[i], y[i] + err[i], y[i] - err[i], y[i]] for i in range(len(y))]
            marks.append(bq.OHLC(x=xerr, y=y_var, marker='candle', scales={'x': x_sc, 'y': y_sc},
                format='ohlc',
        stroke=color[color_count], display_legend=False))

        color_count += 1

    # marks = []
    # if len(lines["y"]) > 0:
    #     marks.append(bq.Lines(x=x, y=lines["y"], scales={'x': x_sc, 'y': y_sc},
    #                 labels=lines["labels"], colors=lines["colors"]))
    # if len(bars["y"]) > 0:
    #     # xe = x.copy()
    #     # ye = np.concatenate(([0], y))
    #     # x, w = edges_to_centers(x)
    #     marks.append(bq.Bars(x=edges_to_centers(x)[0], y=bars["y"], padding=0.0, labels=bars["labels"], opacities=[0.6]*len(x),
    #                colors=[color[color_count]], stroke=color[color_count]))

    fig = bq.Figure(marks=marks, axes=[ax_x, ax_y], title=title)
    tb = bq.Toolbar(figure=fig)

    # ax.set_xlabel(xlab)
    # ax.legend()
    # if title is not None:
        # ax.set_title(title)
    if filename is not None:
        fig.save_png(filename)
    else:
        return VBox((fig, tb))

    return


def __plot_2d(input_data, name=None, axes=None, contours=False, cb=None,
              filename=None, show_variances=False, config=None, title="",
              **kwargs):
    """
    Plot a 2D image.

    If countours=True, a filled contour plot is produced, if False, then a
    standard image made of pixels is created.
    """

    if axes is None:
        axes = input_data.dims

    # Get coordinates axes and dimensions
    coords = input_data.coords
    labels = input_data.labels
    xcoord, ycoord, xe, ye, xc, yc, xlabs, ylabs, zlabs = \
        process_dimensions(input_data=input_data, coords=coords,
                           labels=labels, axes=axes)

    # Parse colorbar
    cbar = parse_colorbar(config.cb, cb)

    # Prepare dictionary for holding key parameters
    data = {"values": {"cbmin": "min", "cbmax": "max", "name": name}}

    if input_data.variances is not None and show_variances:
    #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    #     fig_size = fig.get_size_inches()
    #     fig.set_size_inches(fig_size[0]*2, fig_size[1])
    #     # Append parameters to data dictionary
        data["variances"] = {"cbmin": "min_var", "cbmax": "max_var",
                             "name": "variances"}
    # else:
        # fig, ax = plt.subplots(1, 1)
        # ax = [ax]

    ratio = (xe[-1] - xe[0]) / (ye[-1] - ye[0])
    # fig = plt.figure(title=title,
    #              # layout=Layout(width='650px', height='650px'),
    #              min_aspect_ratio=ratio, max_aspect_ratio=ratio)

    x_sc = bq.LinearScale()
    y_sc = bq.LinearScale()
    # col_sc = bqplot.ColorScale(scheme=cbar["name"])
    # ax_x = bqplot.Axis(scale=x_sc, label='Years')
    # ax_y = bqplot.Axis(scale=y_sc, orientation='vertical', label='Months')
    # ax_c = bqplot.ColorAxis(scale=col_sc)

    # heat = bqplot.GridHeatMap(color=flight_matrix,
    #                scales={'row': x_sc, 'column': y_sc, 'color': col_sc}, stroke='white', row=y.tolist(), column=x.tolist())

    # fig = bqplot.Figure(marks=[heat], axes=[ax_x, ax_y, ax_c],
    #              title="Heatmap of Flight Density from 1949 to 1961", layout=Layout(width='650px', height='650px'))
    # fig


    figs = []
    # col_sc = bq.ColorScale(scheme=cbar["name"])
    ax_x = bq.Axis(scale=x_sc, label=xcoord)
    ax_y = bq.Axis(scale=y_sc, orientation='vertical', label=ycoord)
    # ax_c = bq.ColorAxis(scale=col_sc)
    # ax_c.side = 'right'
    # ax_v = bq.ColorAxis(scale=col_sc)
    # ax_v.side = 'right'


    for i, (key, val) in enumerate(sorted(data.items())):

        # ax[i].set_xlabel(xcoord)
        # ax[i].set_ylabel(ycoord)
        # axes_options = {'x': {'label': xcoord}, 'y': {'label': ycoord}}

        z = getattr(input_data, key)
        # Check if dimensions of arrays agree, if not, plot the transpose
        if (zlabs[0] == xlabs[0]) and (zlabs[1] == ylabs[0]):
            z = z.T
        # Apply colorbar parameters
        if cbar["log"]:
            with np.errstate(invalid="ignore", divide="ignore"):
                z = np.log10(z)
        if cbar[val["cbmin"]] is None:
            cbar[val["cbmin"]] = np.amin(z[np.where(np.isfinite(z))])
        if cbar[val["cbmax"]] is None:
            cbar[val["cbmax"]] = np.amax(z[np.where(np.isfinite(z))])

        # args = {"vmin": cbar[val["cbmin"]], "vmax": cbar[val["cbmax"]],
        #         "cmap": cbar["name"]}

        col_sc = bq.ColorScale(scheme=cbar["name"], min=cbar[val["cbmin"]], max=cbar[val["cbmax"]])
        # ax_x = bq.Axis(scale=x_sc, label=xcoord)
        # ax_y = bq.Axis(scale=y_sc, orientation='vertical', label=ycoord)
        ax_c = bq.ColorAxis(scale=col_sc, label=axis_label(var=input_data, name=val["name"],
                         log=cbar["log"]))

        heat = bq.HeatMap(color=z, x=xc, y=yc, scales={'x': x_sc, 'y': y_sc, 'color': col_sc})

        # heat = bq.GridHeatMap(color=z,
        #                scales={'row': x_sc, 'column': y_sc, 'color': col_sc}, stroke=None) #, row=y.tolist(), column=x.tolist())

        ax_c.side = 'right'
        figs.append(bq.Figure(title=title, marks=[heat], axes=[ax_x, ax_y, ax_c]))
        # fig.marks = [heat]
        # fig.axes = [ax_x, ax_y, ax_c]
        # fig = bq.Figure(marks=[heat], axes=[ax_x, ax_y, ax_c],
        #              title=title)
        #              # min_aspect_ratio=ratio, max_aspect_ratio=ratio) #, layout=Layout(width='650px', height='650px'))
        # fig
        





        # heatmap = plt.heatmap(z, x=xe[1:], y=ye[1:], axes_options=axes_options)
        # if contours:
        #     img = ax[i].contourf(xc, yc, z, **args)
        # else:
        #     img = ax[i].imshow(z, extent=[xe[0], xe[-1], ye[0], ye[-1]],
        #                        origin="lower", aspect="auto", **args)
        # cb = plt.colorbar(img, ax=ax[i])
        # cb.ax.set_ylabel(axis_label(var=input_data, name=val["name"],
                         # log=cbar["log"]))
        # cb.ax.yaxis.set_label_coords(-1.1, 0.5)

    tb = bq.Toolbar(figure=figs[0])

    if filename is not None:
        fig.save_png(filename)
    else:
        # plt.show()
        # return VBox((HBox(figs), tb))
        display(VBox((HBox(figs), tb)))

    return


def get_bqplot_color(index=0):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    return colors[index % len(colors)]
