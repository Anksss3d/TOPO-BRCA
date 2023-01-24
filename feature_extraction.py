import cv2
import numpy as np
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
from skimage import feature

########################################
## Code for Local Persistant Homology ##
########################################


# Betti Curves Function Initializations
CP0 = CubicalPersistence(
    homology_dimensions=[0],
    coeff=3,
    n_jobs=-1
)
CP1 = CubicalPersistence(
    homology_dimensions=[1],
    coeff=3,
    n_jobs=-1
)
CP10 = CubicalPersistence(
    homology_dimensions=[0, 1],
    coeff=3,
    n_jobs=-1
)
BC = BettiCurve(n_bins=100)


def betti0(img):
    '''
    :param img: RGB Image file
    :type img: nd-array
    :return: TDA features vector
    :rtype: nd-array of size (100, )
    '''
    im_gray_h1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diagram_h1_0 =CP0.fit_transform(np.array(img)[None, : , :])
    y_betti_curve = BC.fit_transform(diagram_h1_0)
    return np.reshape(y_betti_curve,100).tolist()


def betti1(img):
    '''
    :param img: RGB Image file
    :type img: nd-array
    :return: TDA features vector
    :rtype: nd-array of size (100, )
    '''
    diagram_h1_0 = CP1.fit_transform(np.array(img)[None, :, :])
    y_betti_curve = BC.fit_transform(diagram_h1_0)
    return np.reshape(y_betti_curve, 100).tolist()

def generate_betti_input_grayscale(img):
    '''
    A  Function to generate a feature vector of size 200 usiing the grayscale value of the image and betti functions.
    :param img: Input image
    :return: 200 Size list of features
    '''
    if len(img.shape) == 3:
        im_gray_h1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        im_gray_h1 = img
    diagram_h1_0 = CP10.fit_transform(np.array(im_gray_h1)[None, :, :])
    y_betti_curves_gray = BC.fit_transform(diagram_h1_0)

    return np.reshape(y_betti_curves_gray, 200).tolist()

