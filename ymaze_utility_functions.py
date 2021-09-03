import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt


def find_centroid(image):
	"""Calculate the centroid coordinates
	of the (possibly thresholded) 'image'
	using the first moments from cv2.
    N.B. result follows row-major conventions.
	"""
    # is this row major or column major????
	M = cv2.moments(image)
	cx = int(M['m10'] / M['m00']) # x position of the center
	cy = int(M['m01'] / M['m00']) # y position of the center
	centroid = [cy, cx] # saved as row major, to work easily with numpy
	return centroid

def ymaze_distances(point, regions):
    """Find the distance between a point (2D vector)
    and a dictionary encoding the discrete positions
    inside the Ymaze. It's quite specific...
    """
    distances = np.zeros((7))
    for i, key in enumerate(regions.keys()):
        distances[i] = np.sqrt((
            point[0] - regions[key][0])**2 \
            + (point[1] - regions[key][1])**2)
    return distances

if __name__ == '__main__':
    x = np.ones((2))
    y = {}
    y[0] = [1, 1]
    y[1] = [2, 2]
    y[2] = [3, 3]

    z = ymaze_distances(x, y)
    print(z)