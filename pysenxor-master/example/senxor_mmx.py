# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019 - 2022. All rights reserved.
#
import sys
import os
import signal
import time
import logging
import argparse
import numpy as np
import cv2 as cv
from pprint import pprint

from senxor.utils import connect_senxor, data_to_frame, remap,\
                         cv_filter, cv_render,\
                         RollingAverageFilter, Display
from senxor.utils import CVSegment
from senxor.plots import Histogram, LinePlot

from imutils.video import VideoStream


np.set_printoptions(precision=1)

# global constants
WHITE = [255]*3
GREEN = [0,255,0]
YELLOW = [0,255,255]
RED = [0,0,255]
DARK = [50]*3
DARKGRAY = [169]*3
VDARK = [3,3,3]
CVFONT = cv.FONT_HERSHEY_SIMPLEX
CVFONT_SIZE = 0.7

HISTO_PARAM = {
    'xlim': (20, 100),
    'ylim': (0,200),
    'bins': 50,
}

LINEPLOT_NFRAMES = 1000
LINEPLOT_PARAM = {
    'xlabel': f'Time, a.u.(last {LINEPLOT_NFRAMES} frames)',
    'ylabel': r'Temperature, $^\circ C$',
    'xlim': (1, LINEPLOT_NFRAMES),
    'ylim': (20, 100),
    'xticks': [],
    'labels': [
        r'$T_{\rm , SX}$',
        r'$T_{\rm , ROI.max}$',
        r'$T_{\rm , HS.mean}$',
        r'$T_{\rm , HS.max}$',
    ],
}

TIP_SEGM_PARAM = {
  # threshold-based segmentation
  # ----------------------------
  # supported: simple, otsu, adaptive
  'threshold_type': 'simple',
  # threshold value for simple thresholding
  'threshold': 190,
  'contour_minArea': -5,

  # contour analysis
  # ----------------
  # absolute value of the area of the smallest contour
  'min_contourarea': 5,
  # extention of the bounding box of the target contour
  # for estimating background temperature
  'bbox_extension': 10,
}

# Make the a global variable and use it as an instance of the mi48.
# This allows it to be used directly in a signal_handler.
global mi48

# define a signal handler to ensure clean closure upon CTRL+C
# or kill from terminal
def signal_handler(sig, frame):
    """Ensure clean exit in case of SIGINT or SIGTERM"""
    logger.info("Exiting due to SIGINT or SIGTERM")
    cv.destroyAllWindows()
    mi48.stop()
    logger.info("Done.")
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tis', '--thermal-image-source', default=None, dest='tis_id',
                        help='Comport name (str) or thermal video source  ID (int)')
    parser.add_argument('-cis', '--color-image-source', type=int, default=None,
                        dest='cis_id',
                        help='Video source ID: 0=laptop cam, 1=USB webcam')
    parser.add_argument('-fps', '--framerate', default=5, type=int, dest='fps',
                        help='Frame rate per second')
    parser.add_argument('-c', '--colormap', default='rainbow2', type=str,
                        help='Colormap for the thermogram')
    parser.add_argument('--data_file', default=None, type=str,
                        help='file instead of camera stream')
    parser.add_argument('-v', '--video-record', default=False, dest='record_video',
                        action = 'store_true', help='Record a video of what is shown')
    parser.add_argument('-e', '--emissivity', type=float, default=0.95,
                        dest='emissivity', help='target emissivity')
    parser.add_argument('-histo', '--show-histogram', default=False, action='store_true',
                        dest='show_histogram', help='Show thermal image histogram')
    parser.add_argument('-plots', '--show-plots', default=False, action='store_true',
                        dest='show_plots',
                        help='Show plots of measured temperatures')
    parser.add_argument('-scale', '--thermal-image-scale-factor', default=4, type=int,
                        dest='img_scale', help='Scale factor for thermogram')

    args = parser.parse_args()
    return args


class VIP():
    """Visual Image Pipeline"""

    def __init__(self, param):
        self.image_size = param['image_size']
        self.ROI = param['ROI']

    def _execute(self, image, input_struct):
        scaled = cv.resize(image, self.image_size)
        # this is the ROI box (diagonal points) in the thermal FPA size
        # in real life this will be the roi of the input image
        images = {
            'input': image,
            'scaled': scaled,
        }
        output_struct = {'scaled_roi': self.ROI}
        return images, output_struct

    def __call__(self, image, input_struct):
        return self._execute(image, input_struct)


class TIP():
    """Thermal Image Pipeline"""

    def __init__(self, param):
        self.image_scale = param.get('image_scale', 1)
        self.colormap = param.get('colormap', 'rainbow2')
        self.ncol_nrow = param.get('fpa_ncol_nrow', (80,62))
        self.image_size = (self.image_scale * self.ncol_nrow[0],
                           self.image_scale * self.ncol_nrow[1])
        self.histogram = None
        if param.get('show_histogram', False):
            self.histogram = Histogram(data=np.zeros(np.prod(self.ncol_nrow)),
                                       figsize=self.image_size,
                                       param=HISTO_PARAM)
        self.lineplot = None
        if param.get('show_plots', False):
            # initialize some dummy data
            # note the +1 to the variable names -- this dimension must cover
            # all variables to be plotted, plus the x-axis variable.
            self.lp_data = np.zeros((LINEPLOT_NFRAMES, len(LINEPLOT_PARAM['labels'])+1))
            # Initialize X-coordinates -- just a frame index is enough
            self.lp_data[:, 0] = list(range(LINEPLOT_NFRAMES))
            self.lineplot = LinePlot(data=self.lp_data,
                                     figsize=self.image_size,
                                     param=LINEPLOT_PARAM)

        # Initialize segmentation object
        self.segment = CVSegment(param)

    def execute(self, frame, input_struct):
        """Thermal data processing pipeline; produces image and stats/metrics"""
        # make up the thermogram
        frame_uint8 = remap(frame)
        self.img_raw = cv_render(frame_uint8, resize=self.image_size,
                                 colormap=self.colormap,
                                 interpolation=cv.INTER_NEAREST, display=False)
        # other processing and analysis go below
        # -------------------------------------

        # ROI
        # -------------------------------------
        # here we assume that the ROI is defined in the coordinates of the
        # upscaled image (rgb) in the format (col,row, wid, hei)
        # note that the input struct may contain no roi
        # in such case,so we take the whole frame to be roi
        scaled_roi = input_struct.get('scaled_roi',
                                      [0, 0, self.ncol_nrow[0], self.ncol_nrow[1]])
        roi = np.rint(np.array(scaled_roi) / self.image_scale).astype('int')
        # roi = frame
        c1, c2 = roi[0], roi[0]+roi[2]-1
        r1, r2 = roi[1], roi[1]+roi[3]-1
        self.roi = frame[r1:r2+1, c1:c2+1]
        roi_stats = {
            'mean': self.roi.mean(),
            'min': self.roi.min(),
            'max': self.roi.max(),
        }

        # SEGMENTATION
        # -------------------------------------
        filtered_ui8 = cv_filter(frame_uint8, parameters={'blur_ks':5},
                             use_median=True, use_bilat=True)
        self.img_filtered = cv_render(filtered_ui8, resize=self.image_size,
                                      colormap=self.colormap,
                                      interpolation=cv.INTER_NEAREST, display=False)
        self.segment(frame=frame, frui8=filtered_ui8)
        # work with hottest segment
        try:
            hs = self.segment.hotspots[0]
            hs_mask = hs.out_frames['hs_mask']
            hs_osd = hs.osd
        except IndexError:
            hs = None
            hs_mask = np.zeros(self.image_size, dtype='uint8')
            hs_osd = {}
        self.img_hs_mask = cv_render(hs_mask, resize=self.image_size,
                                     colormap='parula',
                                     interpolation=cv.INTER_NEAREST, display=False)

        # HISTOGRAMS and LINE PLOTS
        # -------------------------------------
        if self.histogram is not None:
            self.histogram.update(self.roi)
            self.img_histo = self.histogram.get_image()

        if self.lineplot is not None:
            for i, new_y in enumerate([
                    input_struct['Tsx'],
                    roi_stats.get('max', None),
                    hs_osd.get('mean', None),
                    hs_osd.get('max', None),
                ]):
                self.lp_data[:-1, i+1] = self.lp_data[1:, i+1];
                self.lp_data[-1, i+1] = new_y
            self.lineplot.update(self.lp_data)
            self.img_lineplot = self.lineplot.get_image()

        # Compose output
        # -------------------------------------
        images = {
            'raw': self.img_raw,
            'filtered': self.img_filtered,
            'hotspot_mask': self.img_hs_mask,
        }
        if self.histogram is not None:
            images.update({'histogram': self.img_histo,})
        if self.lineplot is not None:
            images.update({'lineplot': self.img_lineplot,})

        output_struct = {
            'roi_min': roi_stats.get('min', None),
            'roi_max': roi_stats.get('max', None),
            'roi_mean': roi_stats.get('mean', None),
        }
        # add stats of hotspot from segmentation
        # this is a problem in that the output struct will retain the
        # previous values until another segmentation succeeds
        if hs is not None:
            output_struct.update(dict([(f'hs_{key}',val)\
                                       for key, val in hs.osd.items()]))
        return images, output_struct

    def __call__(self, thermal_data, input_struct):
        return self.execute(thermal_data, input_struct)


def main():

    # This will enable mi48 logging debug messages
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

    # Define the signals that should be handled to ensure clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse command line arguments
    args = parse_args()

    # Connect and setup thermal camera
    # --------------------------------
    mi48, connected_port, port_names = connect_senxor(src=args.tis_id)
    if mi48 is None:
        logging.critical('Cannot connect to SenXor')
        logging.info(f'The following ports have SenXor attached {port_names}')
        sys.exit(1)
    else:
        logging.info(f'{mi48.sn} connected to {connected_port}')
    logger.info(mi48.camera_info)

    # set up MI48 registers
    # frame rate or frame rate divisor (e.g. mi48.regwrite(0xB4, 10))
    mi48.set_fps(args.fps)
    mi48.regwrite(0xD0, 0x00)
    mi48.disable_filter(f1=True, f2=True, f3=True)
    # enable temperal filter w/ default strength
    mi48.enable_filter(f1=True, f2=False, f3=False)
    # ensure sensitivity enhancement factor is 1.00
    mi48.regwrite(0xC2, 0x64)
    mi48.set_emissivity(args.emissivity)
    mi48.start(stream=True, with_header=True)

    # set rolling average filters for image stabilisation and clipping outliers
    RA_Tmin = RollingAverageFilter(N=10)
    RA_Tmax = RollingAverageFilter(N=10)

    # setup the thermal imaging pipeline (TIP)
    tip_param = {
        'colormap': args.colormap,
        'fpa_ncol_nrow': (mi48.cols, mi48.rows),
        'image_scale': args.img_scale,
        'show_histogram': args.show_histogram,
        'show_plots': args.show_plots,
    }
    tip_param.update(TIP_SEGM_PARAM)
    tip = TIP(tip_param)
    # --------------------------------

    # Connect and setup visual camera
    # --------------------------------
    if args.cis_id is not None:
        vs = VideoStream(src=args.cis_id).start()
        test_frame = vs.read()
    else:
        test_frame = None
    # if we can't get to access to webcam test_frame will be None too
    if test_frame is None:
        vs = None

    # initialize visual imaging pipeline (VIP)
    vip_param = {
        'image_size': tip.image_size,
        # ROI here is given as [(x1, y1, h, w)]
        'ROI': (np.array([[15, 6], [20, 30]]) * tip.image_scale).flatten().tolist(),
    }
    vip = VIP(vip_param)
    # --------------------------------

    # configure the display
    # --------------------------------
    display_options = {
        'window_coord': (0,0),
        'window_title': f'{mi48.camera_id} ({mi48.name}), {args.cis_id}'
    }
    display = Display(display_options)
    # --------------------------------

    # MAIN LOOP
    # -----------------------------------------------------------------
    images = {'visual': {}, 'thermal': {}}
    struct = {'visual': {}, 'thermal': {}}

    while True:

        # grab a visual frame and resize to be the same as thermogram
        # -------------------------------------------------------------
        if vs is not None:
            visual_image = vs.read()
            _imgs, _struct = vip(visual_image, input_struct=None)
            images['visual'].update(_imgs)
            struct['visual'].update(_struct)

        # grab a frame from thermal camera
        # -------------------------------------------------------------
        raw_data, header = mi48.read()
        frame = data_to_frame(raw_data, (mi48.cols, mi48.rows), hflip=True)
        # update min/max Rolling Average values and clip data
        Tmin, Tmax = RA_Tmin(frame.min()), RA_Tmax(frame.max())
        frame = np.clip(frame, Tmin, Tmax)
        # process the thermal frame and return an image
        _imgs, _struct,  = tip(frame, input_struct=\
            {**struct['visual'], 'Tsx': header['senxor_temperature']})
        images['thermal'].update(_imgs)
        struct['thermal'].update(_struct)

        # select rendered images, annotate and display them
        # -------------------------------------------------------------
        display_images = []
        if vs is not None:
            display_images.append(images['visual']['scaled'])
        display_images.append(images['thermal']['raw'])
        display_images.append(images['thermal']['filtered'])
        #    display_images.append(images['thermal']['filtered'])
        if args.show_histogram:
            display_images.append(images['thermal']['histogram'])
        if args.show_plots:
            display_images.append(images['thermal']['lineplot'])
        display_images.append(images['thermal']['hotspot_mask'])

        # the following may need to become more explicit
        for img in display_images[0:2]:
            try:
                cv.rectangle(img, struct['visual']['scaled_roi'], GREEN, 1)
            except KeyError:
                # no ROI from visual stream
                pass

        # display on screen
        display(display_images)

        # handle any keyboard events
        # -------------------------------------------------------------
        # cv.waitKey returns -1 if no key pressed within the given time
        key = cv.waitKey(1)  # & 0xFF
        if key != -1:
            # key was pressed
            if key in [ord("q"), 27]:
                mi48.stop()
                cv.destroyAllWindows()
                if vs is not None:
                    vs.stop()
                break
            else:
                # potentially, put here a call to keyboard handler
                # for some interaction with user
                continue
    # --------------------------------


if __name__ == '__main__':
    main()
