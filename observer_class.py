import numpy as np
import matplotlib.pyplot as plt
import ymaze_utility_functions as utils
from picamera import PiCamera
import time
import cv2

# n.b. row-major notation is user everywhere.

class Observer(object):
    """Control the camera; can loop through continuous acquisitions,
    while finding the larva position
    and assigning it to a discrete set of states (seven states)
    defined by the user.
    """
    def __init__(
        self,
        fps=15,
        resolution=[320, 240], #width, height
        threshold = 10
        ):
        """Give:
        - fps = frames per second
        - resolution = [rows, cols], in pixels
            - there may be a bit of distortion, since the original
                resolution is (1920, 1440), so not a square.
                But we can assume this is minimal and not affecting
                the tracking: we don't care about the exactly exact shape
                of things, right?
        - threshold = minimal distance between current and next position
            (if smaller that this, it's not considered as real transitio)
            (this value is also the diamter of the circles
            drawn when selectiong states)
        """
        super().__init__()
        self.fps = fps
        self.camera = PiCamera(resolution=resolution, framerate=fps)
        # take a first image to check the camera is working
        self.image = np.empty((resolution[1], resolution[0], 3), dtype=np.uint8)
        self.camera.capture(self.image, 'rgb')
        print('Able to take an image of dims: (rows, cols, colors)', self.image.shape)
        self.threshold = threshold
        self.regions = {}
        self.previous_state = 0
        self.state = 0
        self.spot = [1, 1] # positions are all positive integers.
        # this acts as an extra flag to signal that the larva was not found
        self.mean_background = 0
        self.found_flag = False
        # the regions are numbered according to a hard-coded convention
        # it makes it easier to identify where the larva is.
        # see schematics for explanation
        self.regions_keys = [0, 1, 2, 3, 5, 7, 9]
        for key in self.regions_keys:
            self.regions[key] = [0, 0]
        self.count = 0 # iterator used specifically for regions selection
        self.first_state_flag = True # fist time a state is found,
                                    # previous and current state become the same
        self.transition_flag = False
        self.transition = 100 # start with dummy value, before any transition is actually found

        # Two dictionaries in case states and transitions may need to be translated
        # into more understandable words. Feed them the state and the transition
        #respectively and find what is happening.

        self.state_translator = {}
        self.state_translator[0] = ['center']
        self.state_translator[1] = ['\ right channel']
        self.state_translator[2] = ['| top channel']
        self.state_translator[3] = ['/ left channel']
        self.state_translator[5] = ['right circle']
        self.state_translator[7] = ['top circle']
        self.state_translator[9] = ['left circle']

        self.transition_translator = {}
        self.transition_translator[1] = ['center to right']
        self.transition_translator[-1] = ['right to center']

        self.transition_translator[2] = ['center to top']
        self.transition_translator[-2] = ['top to center']

        self.transition_translator[3] = ['center to left']
        self.transition_translator[-3] = ['left to center']

        self.transition_translator[4] = ['right channel to right circle']
        self.transition_translator[-4] = ['right circle to right channel']

        self.transition_translator[5] = ['top channel to top circle']
        self.transition_translator[-5] = ['top circle to top channel']

        self.transition_translator[6] = ['left channel to left circle']
        self.transition_translator[-6] = ['left circle to left channel']

        self.transition_translator[100] = ['Stating value dumped']

    def compute_background(
        self,
        time_window=10
        ):
        """Calculate the background
        taking images for 'time_window' seconds
        """
        now = time.time()
        start = now
        elapsed = now - start
        prev_image_time = start
        image_time = now - prev_image_time

        self.background = []
        self.camera.capture_continuous(self.image, 'rgb')
        while elapsed < time_window:
            if(image_time >= 1 / self.fps):
                self.background.append(self.image)
                prev_image_time = time.time()
            now = time.time()
            image_time = now - prev_image_time
            elapsed = now - start
        self.background = np.asarray(self.background)
        self.background = np.mean(self.background, axis = 0)
        self.background = np.mean(self.background, axis = 2)

    def define_states(
        self,
        ):
        self.click_me = plt.figure('Select regions')
        self.ax = self.click_me.add_subplot(111)
        self.ax.imshow(self.background)
        cid = self.click_me.canvas.mpl_connect('button_press_event', self.select_rois)
        plt.show()

    def select_rois(
        self,
        event
        ):
        """Capture click event in matplotlib and register its position
        n.b. coordinates inverted to conform to "row major" notation (np)
        Better annotations may be added (e.g. a circle)
        """
        self.regions[self.regions_keys[self.count]] = [int(event.ydata), int(event.xdata)]
        # next line, from cartesian to row-major notation
        self.ax.annotate(str(self.regions_keys[self.count]), xy=(event.xdata+5, event.ydata-5))
        self.ax.plot(
            event.xdata,
            event.ydata,
            'o',
            ms=self.threshold,
            mec='b',
            mfc='none',
            mew=2
            )
        self.click_me.canvas.draw()

        self.count += 1
        if self.count > 6:
            self.count = 0
            self.ax.clear()
            self.ax.imshow(self.background)

    def find_larva(
        self,
        lowpass_kernel=np.ones((5, 5)),
        highpass_kernel=np.ones((3, 3))
        ):
        """Find larva's centroid.
        TO BE TESTED!
        """
        self.camera.capture(self.image).array
        self.image = cv2.subtract(self.image, self.mean_background)
        # threshold, then low-pass, then high-pass
        _, self.thresholded = cv2.threshold(
            self.image,
            self.threshold,
            255,
            cv2.THRESH_BINARY
            )
        self.thresholded = cv2.morphologyEx(
            self.thresholded,
            cv2.MORPH_CLOSE,
            lowpass_kernel
            )
        self.thresholded = cv2.morphologyEx(
            self.thresholded,
            cv2.MORPH_OPEN,
            highpass_kernel
            )
        # find the contours. We assume the only one
        # (or at lest the largest one) is the larva
        self.contours, _ = cv2.findContours(
            self.thresholded, # input image
            cv2.RETR_TREE, # contour retrieaval mode
            cv2.CHAIN_APPROX_NONE # contour approximation method
		)
        areas = []
        for contour in self.contours:
            areas.append(cv2.contourArea(contour))
        # calculate the centroid
        if len(areas) >= 1:
            self.larva = self.contours[np.argmax(areas)]
            self.spot = utils.frind_centroid(self.larva)
            self.found_flag = True
        else:
            self.spot = [-1, -1]
            self.found_flag = False

    def get_state(self):
        """Find the minimal distance between the locations
        and the given point. Store the index in self.state"""
        # TO BE TESTED
        if self.spot == [-1, -1]:
            pass # do nothing if larva centre is not properly found
        else:
            # instantaneous distance between now and the previous state-point
            my_distance = np.linalg.norm(np.asarray(self.spot) - np.asarray(self.regions[self.state]))
            # distances between now and all state-points
            n_distances = utils.ymaze_distances(
                self.spot,
                self.regions
                )
            closest = np.amin(n_distances)
            if ((my_distance - self.threshold) > closest
            and closest != my_distance):
                self.previous_state = self.state # keep track of previous state
                self.state =  list(self.regions.keys())[
                            np.argmin(n_distances)
                            ]
                self.transition_flag = True
            if self.first_state_flag == True:
                self.previous_state = self.state
                self.first_state_flag = False
                self.transition_flag = False
        # the state can create send out a signal

    def monitor_transition(self):
        """This is where the transition
        between the discrete states is computed.
        The hardcoded numbers follow a naming logic explained in schematics.
        For now, the important transitions are just after a decision
        (to eventually give immediate reward/punishment)
        and the ones after reaching a new circle
        (to change the flows)
        """
        if self.transition_flag == False:
            pass
        else:
            self.transition = self.state - self.previous_state
            self.transition_flag = False
        # actually from now on are computations to be done externally
        # all the following prints can be deleted
            if (np.abs(self.transition) <= 3):
                # choice has been made.
                # eventually punish or reward accordingly
                # and change flows
                # using from "manipulation_classes.py"
                if self.transition == -1:
                    print('going away from the center')
                if self.transition == -2:
                    print('going away from the center')
                if self.transition == -3:
                    print('going away from the center')
                else:
                    print('going away from the center')
            elif (np.abs(self.transition) > 3 and np.abs(self.transition) <=6):
                # do something to the valves
                if self.transition == 4:
                    print('going to bottom right circle')
                if self.transition == 5:
                    print('going to top circle')
                if self.transition == 6:
                    print('going to bottom left circle')


if __name__ == '__main__':

    test = Observer() # OK

    test.compute_background(1) # OK
    test.define_states()
    
    test.get_state()

