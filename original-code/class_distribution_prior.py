import scipy.linalg as la
# for linear algebra

import numpy as np
# for some maths functions

from scipy.sparse import bmat, csc_matrix
# bmat: for constructing the block matrix in the MAP saddle point system
# csc_matrix: for filling holes in the MAP saddle point system

import scipy.sparse.linalg as sla
# for solving

import warnings
# for warning, because raising them causes the program to stop.


class PriorDistribution:
    def __init__(self, mean, covar, covar_inv=None):
        """
        initialization of gaussian prior

        # TODO: at the moment, we only consider small parameter spaces in which it should be ok to compute the
        # TODO: inverse of the prior covariance matrix. At some point we should change this though.

        @param mean: mean
        @param covar: covariance matrix (symmetric positive definite)
        @param covar_inv: inverse covariance matrix
        """
        self.mean = mean
        self.covar = covar
        self.nPara = covar.shape[0]

        if covar_inv is None:
            self.covar_inv = la.inv(covar)
        else:
            self.covar_inv = covar_inv

        self.covarInvMean = self.covar_inv.dot(self.mean)

    def apply_covarInv(self, para):
        """
        This function applies the inverse of the prior covariance matrix to a given parameter.

        @param para: parameter
        @return: Covar^-1 * para
        """
        return self.covar_inv.dot(para)

    def compute_norm2(self, para):
        """
        This function computes the squared norm of the given parameter in the norm induced by inverse of the prior
        covariance matrix.

        @param para: parameter of which to compute the norm
        @return: squard norm:
        """
        return para.T.dot(self.covar_inv.dot(para))

