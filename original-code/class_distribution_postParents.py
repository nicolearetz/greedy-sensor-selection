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

from class_distribution_postGrandparents import PosteriorGrandparents

# TODO: not sure if this class structure makes the best sense this way...
# TODO: maybe better with classmethod()?


class PosteriorParents:

    def __init__(self, model, prior, sensorset, sigma2):
        self.sensorset = sensorset
        self.sigma2 = sigma2
        self.parent = PosteriorGrandparents(model, prior)

    def compute_postCovarInv(self, theta, **kwargs):
        """
        calls the computation method for the inverse posterior covariance matrix in the posteriorCovInv class with
        its own arguments

        @param theta:
        @param kwargs:
        @return:
        """
        return self.parent.compute_postCovarInv(self.sensorset, theta, self.sigma2, **kwargs)

    def compute_eigvals(self, theta, **kwargs):
        """
        returns a decreasing array of the eigenvalues of the posterior covariance matrix

        @param theta:
        @param kwargs:
        @return:
        """
        eigvals = la.eigh(self.compute_postCovarInv(theta, **kwargs), eigvals_only=True)
        return 1.0/eigvals

    def compute_obsCoeff(self, theta):
        """
        computes the observability coefficient at the given hyper-parameter

        @param theta: hyper-parameter
        @return:
        """
        Y_hyper = self.model.forwardsolve_all(theta)
        SP_hyper = Y_hyper.T.dot(self.model.SP.dot(Y_hyper))
        SP_noise, __, __ = self.sensorset.compute_SPnoise(Y_hyper)

        try:
            SP_hyper.shape[0]
        except IndexError:
            # catch the case in which the parameter-space is one dimensional
            return np.array([np.sqrt(SP_noise / SP_hyper)]), np.array([1.0 / np.sqrt(SP_hyper)])

        eigval, eigvec = la.eigh(SP_noise, SP_hyper, eigvals=[0, 0])

        if SP_noise.shape[0] < SP_hyper.shape[0]:
            # less sensors than parameters -> observability coefficient equals zero
            eigval = np.array([0])

        return np.sqrt(np.abs(eigval[0])), eigvec

    def MAP_mupy(self, theta, measured, **kwargs):
        """
        This function computes the map point parameter (m), its associated state = solution of the forward problem (u),
        and adjoint variable = solution of the adjoint problem (p).

        @param sensorset: sensors
        @param theta: hyper-parameter
        @param sigma2: scaling
        @param measured: measurements
        @param Lq: see fct compute_Lq
        @param kwargs: might specify MAP routine
        @return: MAP point, assoc. state, associated adjoint
        """
        return self.parent.MAP_mupy(self.sensorset, theta, self.sigma2, measured, **kwargs)

    def compute_norm2_mupy(self, mupy):
        """
        This functions computes the squared norms of the different solution parts in the mupy variable.
        It is not in the posterior grandparents class because the mupy is strongly connected to the sensorset it
        was created with.

        @note: with this weird class structure I don't think it makes much of a difference anyway...

        @param mupy:
        @return:
        """
        norm_m = self.prior.compute_norm2(mupy[0])
        norm_u = self.model.compute_norm2(mupy[1])
        norm_p = self.model.compute_norm2(mupy[2])
        norm_y = self.sensorset.compute_norm2(mupy[3])

        return norm_m, norm_u, norm_p, norm_y

    def MAP_directFormula(self, theta, measured, **kwargs):
        """
        This function computes the posterior MAP point with the direct formula
        map = post. covariance matrix * (adj. parameter-to-observable map applied to inverse noise covariance matrix
                                                applied to measurement data over sigma squared
                                            + prior covariance matrix inverse applied to prior mean)

        @param theta:
        @param measured:
        @param kwargs:
        @return:
        """
        return self.parent.MAP_directFormula(self.sensorset, theta, self.sigma2, measured, **kwargs)

    def MAP_saddlepoint(self, theta, measured, **kwargs):
        """
        this function computes the MAP point by solving the saddle point problem.

        @param theta: hyper-parameter
        @param measured: measurements
        @param kwargs:
        @return:
            [0]: MAP-point
            [1]: state at MAP-point
            [2]: adjoint state at MAP-point
            [3]: difference in measurements
        """
        return self.parent.MAP_saddlepoint(self.sensorset, theta, self.sigma2, measured, **kwargs)

    def adjointsolve(self, theta, measured=None, state=None, **kwargs):
        """
        this function computes the solution of the adjoint equation in the MAP minimization problem for a linear
        forward model of type Au = Bm (u: state, m: parameter).

        This function is not in the forward model classes FEmodel or the RBmodel class, because adjoint equations are
        defined through the minimization problems that shall be solved. The forward model classes don't know about
        any minimization.

        If either no state or no measurements are given, the function only uses the other, given one.
        If neither is given, it uses all possible combinations of measurements (this corresponds to the case where
        measured it the nSensor x nSensor identity matrix)

        @param theta: hyper-parameter
        @param measured: measurements
        @param state: state vector, or basis of a subspace
        @param kwargs:
        @return:
        """
        return self.parent.adjointsolve(self.sensorset, theta, self.sigma2, measured, state, **kwargs)

    @property
    def model(self):
        return self.parent.model

    @property
    def prior(self):
        return self.parent.prior
