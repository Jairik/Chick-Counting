# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019 - 2022. All rights reserved.
#
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import cv2 as cv
matplotlib.use('TkAgg')
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_hist_patch(data, *args, **kwargs):
    """Calculate counts and bins and return a patch for drawing"""

    # extract some args for the rendering of histogram
    # the rest -- pass to np.hystogram
    hist_edge_color = kwargs.pop('hist_edge_color', 'yellow')
    hist_face_color = kwargs.pop('hist_face_color', 'green')
    hist_face_alpha = kwargs.pop('hist_face_alpha', 0.5)

    # get histogram data: counts and bin values
    counts, bins = np.histogram(data, *args, **kwargs)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + counts
    nrects = len(left)

    # Generate paths that can be visualised by an artist
    # The path must be defined in terms of vertexes and codes (c.f. matplotlib.path)
    # The relevant codes here are: LINETO, MOVETO, CLOSEPOLY, because
    # we're dealing with rectangular patches, one per bin
    # For each rectangle, we need to:
    # * move to left,bottom
    # * draw three lines in desired colour, and
    # * close the polygon.
    # This is equivalent to having 5 vertexes that define the path
    nverts = nrects * 5
    # Each virtex has 2 coordinates: (x,y), hence a 2D array is needed
    # and each virtex requires an integer code:
    # Use stride of 5 to address corresponding vertex and code of each rectangle
    verts = np.zeros((nverts, 2))
    verts[0::5, 0] = left
    verts[0::5, 1] = bottom
    verts[1::5, 0] = left
    verts[1::5, 1] = top
    verts[2::5, 0] = right
    verts[2::5, 1] = top
    verts[3::5, 0] = right
    verts[3::5, 1] = bottom
#    verts[4::5] = verts[0::5]
    codes = np.ones(nverts, dtype=np.uint8) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY

    # compose that line paths and enclosed patches
    _path = path.Path(verts, codes)
    patch = patches.PathPatch(_path, facecolor=hist_face_color,
                              edgecolor=hist_edge_color, alpha=hist_face_alpha)
    return patch

def get_image(figure):
    """Transform a matplotlib `figure` to an image to be displayed by `opencv.imshow`"""
    # redraw the canvas
    canvas = figure.canvas
    canvas.draw()
    # convert canvas to image
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    # matplotlib img is rgb, convert to opencv's default bgr
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return img

class Histogram:
    """Continuously updatable histogram plot"""

    def __init__(self, data, figsize=(6,5), param=None):
        # create a figure to be updated
        if figsize[1] > 20:
            # assume figsize is given in pixels if heigth > 20
            # to translate to inches, we need the screen DPI
            try:
                self.dpi = param.get('dpi', 100)
            except TypeError:
                # param is None
                self.dpi = 100
            figsize = [x / self.dpi for x in figsize]
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        self.fig.tight_layout()
        if param.get('xlabel', None) is not None:
            self.ax.set_xlabel(param.pop('xlabel'))
        if param.get('ylabel', None) is not None:
            self.ax.set_ylabel(param.pop('ylabel'))
        self.ax.autoscale(enable=False)
        if param.get('xlim', None) is not None:
            self.ax.set_xlim(param.pop('xlim'))
        if param.get('ylim', None) is not None:
            self.ax.set_ylim(param.pop('ylim'))
        if param.get('xticks', None) is not None:
            self.ax.xaxis.set_ticks(param.pop('xticks'))
        if param.get('yticks', None) is not None:
            self.ax.yaxis.set_ticks(param.pop('yticks'))
        self.labels = param.pop('labels', None)
        self.data = data
        self.patch = get_hist_patch(data, **param)
        self.nbins = param.get('bins', 50)
        self.ax.add_patch(self.patch)

    def update(self, data=None):
        if data is not None:
            self.data = data
        patch = get_hist_patch(self.data, bins=self.nbins)
        self.patch.set_path(patch.get_path())
        # return list is required by FuncAnimation
        return self.patch,

    def get_image(self):
        """Return an image to be displayed by `OpenCV.imshow`"""
        # redraw the canvas
        return get_image(self.ax.figure)


class LinePlot():
    """Lie plot of one or more variables on same Y axis"""
    def __init__(self, data, figsize=(6,5), param=None):
        # create a figure to be updated
        if figsize[1] > 20:
            # assume figsize is given in pixels if heigth > 20
            # to translate to inches, we need the screen DPI
            try:
                self.dpi = param.get('dpi', 100)
            except TypeError:
                # param is None
                self.dpi = 100
            figsize = [x / self.dpi for x in figsize]
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        self.ax.autoscale(enable=True)
        if param.get('xlabel', None) is not None:
            self.ax.set_xlabel(param['xlabel'])
        if param.get('ylabel', None) is not None:
            self.ax.set_ylabel(param['ylabel'])
        if param.get('xlim', None) is not None:
            self.ax.set_xlim(param['xlim'])
        if param.get('ylim', None) is not None:
            self.ax.set_ylim(param['ylim'])
        if param.get('xticks', None) is not None:
            self.ax.xaxis.set_ticks(param['xticks'])
        if param.get('yticks', None) is not None:
            self.ax.yaxis.set_ticks(param['yticks'])
        # for convenience put grid and labels on each side of y-axis
        # self.ax.tick_params(labelright=True, right=True)
        self.ax.grid(True)
        self.ax.figure.tight_layout()
        self.labels = param.get('labels', None)
        # establish a reference to a data object that
        # is updated outside, but is accessible to self.update
        self.data = data
        # define the plot object
        self.markers = ['x', '+', 'x', '+', 's', 'd']
        self.markers = ['+']*6
        self.setup(nvars=self.data.shape[1]-1)

    def setup(self, nvars=1):
        """Setup an empty scatter plot which will be dynamically updated"""
        self.lines = []
        for i in range(nvars):
            lines = self.ax.plot(self.data[:, 0], self.data[:, i+1],
                                 marker=self.markers[i], ms=3, linestyle='None')
            self.lines.append(lines[0])
        # print('Initilizing plot')
        # print(self.lines, len(self.lines))
        if self.labels is not None:
            self.ax.legend(self.lines, self.labels, loc=3, ncol=2)
        self.fig.tight_layout()
        return self.lines,

    def update(self, data=None):
        """Update the positions, colors and size of the scatter points"""
        #t0 = time.time()
        if data is None:
            data = self.data
        for i, line in enumerate(self.lines):
            # update only ydata, xdata is set in self.setup()
            line.set_ydata(data[:, i+1])
        #cost = time.time() - t0
        #print('Plot update: {} ms'.format(cost*1.e3))
        return self.lines,

    def get_image(self):
        """Return an image to be displayed by `OpenCV.imshow`"""
        # redraw the canvas
        return get_image(self.ax.figure)


class LivePlot2Y():
    """Live plot of variables on 2 Y-axis"""
    def __init__(self, data, data2, figsize=(6,5), param=None):
        # create a figure to be updated
        if figsize[1] > 20:
            # assume figsize is given in pixels if heigth > 20
            # to translate to inches, we need the screen DPI
            try:
                self.dpi = param.get('dpi', 100)
            except TypeError:
                # param is None
                self.dpi = 100
            figsize = [x / self.dpi for x in figsize]
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1,
                               figsize=figsize)
        self.ax.autoscale(enable=True)
        self.ax2 = self.ax.twinx()
        # X axis
        if param.get('xlabel', None) is not None:
            self.ax.set_xlabel(param['xlabel'])
        if param.get('xlim', None) is not None:
            self.ax.set_xlim(param['xlim'])
        if param.get('xticks', None) is not None:
            self.ax.xaxis.set_ticks(param['xticks'])
        # Y axis (left)
        if param.get('ylabel', None) is not None:
            self.ax.set_ylabel(param['ylabel'])
        if param.get('ylim', None) is not None:
            self.ax.set_ylim(param['ylim'])
        if param.get('yticks', None) is not None:
            self.ax.yaxis.set_ticks(param['yticks'])
        # Y2 axis (right)
        if param.get('y2label', None) is not None:
            self.ax2.set_ylabel(param['y2label'])
        if param.get('y2lim', None) is not None:
            self.ax2.set_ylim(param['y2lim'])
        if param.get('y2ticks', None) is not None:
            self.ax2.yaxis.set_ticks(param['y2ticks'])
        # Line labels and colors
        self.labels = param.get('labels', None)
        self.colors = param.get('colors', None)
        # Establish a reference to a data object that
        # is updated outside, but is accessible to self.update
        # Note: data has X and left-Y items, data2 has only right-Y items
        self.data = data
        self.data2 = data2
        # define the plot object
        self.setup(nvars=self.data.shape[1]-1,
                   nvars2=self.data2.shape[1])
        self.fig.tight_layout()

    def setup(self, nvars=1, nvars2=1):
        """Setup an empty scatter plot which will be dynamically updated"""
        self.lines = []
        self.lines2 = []
        # ax: note that data contains X values as first column
        for i in range(nvars):
            lines = self.ax.plot(self.data[:, 0], self.data[:, i+1],
                                 marker='o', ms=3, linestyle='None')
            self.lines.append(lines[0])
        # ax2: data2 contains only Y values; reuse X from data
        for i in range(nvars2):
            lines = self.ax2.plot(self.data[:, 0], self.data2[:, i],
                                  marker='+', ms=5, linestyle='None')
            self.lines2.append(lines[0])
        if self.labels is not None:
            self.ax.legend(self.lines, self.labels[:len(self.lines)], loc=2)
            self.ax2.legend(self.lines2, self.labels[len(self.lines):], loc=1)
        if self.colors is not None:
            assert len(self.colors) == len(self.lines) + len(self.lines2)
            for i, line in enumerate(self.lines):
                line.set_color(self.colors[i])
            for i, line in enumerate(self.lines2):
                line.set_color(self.colors[i+len(self.lines)])
        self.ax.legend(self.lines, self.labels[:len(self.lines)], loc=2)
        self.ax2.legend(self.lines2, self.labels[len(self.lines):], loc=1)
        return self.lines, self.lines2

    def update(self, *args, **kwargs):
        """Update the positions, colors and size of the scatter points"""
        #t0 = time.time()
        try:
            data = kwargs['data']
            data2 = kwargs['data2']
        except KeyError:
            data = self.data
            data2 = self.data2
        for i, line in enumerate(self.lines):
            # update only ydata, xdata is set in self.setup()
            line.set_ydata(data[:, i+1])
        for i, line in enumerate(self.lines2):
            # update only ydata, xdata is set in self.setup()
            line.set_ydata(data2[:, i])
        # must return a list of artists to use 'blit=True'
        #cost = time.time() - t0
        #print('Plot update: {} ms'.format(cost*1.e3))
        return self.lines, self.lines2

    def get_image(self):
        """Return an image ready to be displayed by OpenCV"""
        # redraw the canvas
        canvas = self.ax.figure.canvas
        canvas.draw()
        # convert canvas to image
        img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        # matplotlib img is rgb, convert to opencv's default bgr
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        return img


