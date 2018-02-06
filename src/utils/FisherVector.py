#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
from data_utils import ReadImage, rgb_normalized


def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

   # kpimg = cv2.drawKeypoints(im_gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
   # plt.imshow(kpimg)

def computeFV(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.

    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors

    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.

    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.

    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf

    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]
    
    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
            - Q_xx_2
            - Q_sum * gmm.means_ ** 2
            + Q_sum * gmm.covariances_
            + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))



def FeatureExtract(im_file, filt, nkeys, pca, gmm, scaler):
    # read image
    im = ReadImage(im_file)
    
    # rgb normalization
   # im = rgb_normalized(im)
    
    # to gray
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #im_gray = cv2.normalize(im_gray,im_gray, 0, 255, cv2.NORM_MINMAX)

    # apply gaussian filter
    #im_gray = cv2.GaussianBlur(im_gray,(15,15),0)
    
    # extract SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = nkeys)
    kp, descriptors = sift.detectAndCompute(im_gray, None)

    descriptors /= (descriptors.sum(axis=1, keepdims=True) + 1e-7)
    descriptors = np.sqrt(descriptors)
    
    # apply pca transform
    descriptors = pca.transform(descriptors)
    
    # compute Fisher Vector
    fv = computeFV(descriptors, gmm)
    
    # power-normalization
   # fv = np.sign(fv) * np.abs(fv) ** 0.5
    # L2 normalize
    #fv /= np.sqrt(np.sum(fv ** 2))
    
    return fv
