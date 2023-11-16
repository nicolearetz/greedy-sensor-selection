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
warnings.filterwarnings(action="always")
# for warning, because raising them causes the program to stop.

# TODO: not sure if this class structure makes the best sense this way...


class PosteriorGrandparents:

    def __init__(self, model, prior):
        """
        initialization of the outermost posterior distribution class. It knows its prior, but that's about it.
        It will be married to a sensorset, which gives the posterior_parents class. Married with concrete data,
        this gives then the posterior_distribution class.

        @param prior: prior distribution
        @param fom: fullorder model
        """
        self.prior = prior
        self.model = model

    def compute_postCovarInv(self, sensorset, theta, sigma2, **kwargs):
        """
        computes the posterior covariance matrix at a given hyper-parameter, with a given sensorset, and noise variance
        scaling factor.

        @param sensorset:
        @param theta:
        @param sigma2:
        @param kwargs:
        @return:
        """

        Y_hyper = self.model.forwardsolve_all(theta)
        SP_noise, SP_noise_factor, para2obs = sensorset.compute_SPnoise(Y_hyper)
        covarInv = SP_noise / sigma2 + self.prior.covar_inv

        if kwargs.get("bool_returnIntermediateInfo", False):
            return covarInv, Y_hyper, SP_noise, SP_noise_factor, para2obs
        else:
            return covarInv

    def compute_para2obs(self, sensorset, theta, **kwargs):
        """
        this function computes the parameter-to-observable map.

        @param sensorset:
        @param theta:
        @param kwargs:
        @return:
        """
        Y_hyper = self.model.forwardsolve_all(theta)
        return sensorset.measure(Y_hyper)

    def MAP_mupy(self, sensorset, theta, sigma2, measured, **kwargs):
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
        routine = kwargs.get("MAP_routine", "direct")

        if routine == "direct":
            m = self.MAP_directFormula(sensorset, theta, sigma2, measured, **kwargs)  # map point
            u = self.model.forwardsolve(theta, m)  # state
            y = sensorset.measure(u).T  # measurements
            p = self.adjointsolve(sensorset, theta, sigma2, measured=measured - y)  # adjoint
            return m, u, p, y

        if routine == "minimization" or routine == "min":
            return self.MAP_minimization(sensorset, theta, sigma2, measured, **kwargs)

        if routine == "saddlepoint" or "saddle-point" or "sp":
            return self.MAP_saddlepoint(sensorset, theta, sigma2, measured, **kwargs)

        raise KeyError("In MAP_mup: invalid keyword for MAP routine.")

    def MAP_directFormula(self, sensorset, theta, sigma2, measured, **kwargs):
        """
        This function computes the posterior MAP point with the direct formula
        map = post. covariance matrix * (adj. parameter-to-observable map applied to inverse noise covariance matrix
                                                applied to measurement data over sigma squared
                                            + prior covariance matrix inverse applied to prior mean)

        @param sensorset:
        @param theta:
        @param sigma2:
        @param measured:
        @param kwargs:
        @return:
        """

        # compute first summand in the direct formula
        temp = kwargs.get("bool_returnIntermediateInfo", None)
        kwargs["bool_returnIntermediateInfo"] = True
        covarInv, __, __, __, para2obs = self.compute_postCovarInv(sensorset, theta, sigma2, **kwargs)
        if temp is None:
            kwargs.pop("bool_returnIntermediateInfo")
        else:
            kwargs["bool_returnIntermediateInfo"] = temp

        rhs = la.solve(sensorset.covar, measured)
        rhs = para2obs.T.dot(rhs) / sigma2

        # add second summand
        rhs = rhs + self.prior.covarInvMean

        # apply posterior covariance matrix
        map = la.solve(covarInv, rhs)
        return map

    def MAP_minimization(self, sensorset, theta, sigma2, measured, **kwargs):
        """
        this function computes the MAP point by solving the minimization problem

        @param sensorset:
        @param theta:
        @param sigma2:
        @param measured:
        @param kwargs:
        @return:
        """
        raise NotImplementedError("This function hasn't been implemented yet...")

    def MAP_saddlepoint(self, sensorset, theta, sigma2, measured, **kwargs):
        """
        this function computes the MAP point by solving the saddle point problem.

        Note: one could think that it might be easier to first compute
        Y_hyper = self.fom.forwardsolve_all(theta)
        and then project the saddle point equations onto this space. However, this will not work:
        1.) the test space of the forward problem still needs to account for stability, so we need the supremizers
        here (unless the problem is coercive, then a minimum stability of the forward problem is guaranteed)
        2.) for the solution of the saddle-point problem to still approximate the original solution of the minimization,
        the adjoint space needs to include the solution space of the adjoint equation
        3.) this makes the test space in total twice as big as the trial space
        It is still an open problem how to guarantee both stability and good approximation qualities of the minimization
        solution for reduced order models (and the idea here of projecting onto Y_hyper corresponds to an RB model
        reduction)

        @param Lq: array of size (3,)
            Lq[0]: (inverse noise covariance operator) applied to L, shape (nSensors, model.nDim)
            Lq[1]: observation operator L, shape (nSensors, model.nDim)
            Lq[2]: L.T.dot(Lq[1]), shape (model.nDim, model.nDim)
        @param theta: hyper-parameter
        @param sigma2: sigma squared
        @param measured: measurements
        @param kwargs:
        @return:
            [0]: MAP-point
            [1]: state at MAP-point
            [2]: adjoint state at MAP-point
        """

        A_theta, B_theta = self.model.assemble(self.model.Aq, self.model.Bq, theta)
        rhs_adj = self.model.apply_BC_zeroDirichlet(sensorset.Lq[0].dot(measured))

        if self.model.bool_sparseSolvers:

            # set up left-hand-side matrix (symmetric saddle-point matrix)
            LHS = bmat([
                [self.prior.covar_inv, None, -B_theta.T / sigma2],
                [None, sensorset.Lq2, A_theta.T],
                [-B_theta, A_theta, None]
            ], format="csc")

            # set up right-hand-side vector
            RHS = bmat([
                [csc_matrix(self.prior.covarInvMean).T],
                [np.array([rhs_adj]).T],
                [csc_matrix((self.model.nDim, 1))]
            ], format="csc")

            # solve linear system
            v = sla.spsolve(LHS, RHS)
            # TODO: For some reason, it takes way too long to use an iterative solver here. Probably because the
            # TODO: preconditioning is unclear.

        else:
            # set up left-hand-side matrix (symmetric saddle-point matrix)
            LHS = np.block([
                [self.prior.covar_inv, np.zeros((self.model.nPara, self.model.nDim)), -B_theta.T / sigma2],
                [np.zeros((self.model.nDim, self.model.nPara)), sensorset.Lq2, A_theta.T],
                [-B_theta, A_theta, np.zeros((self.model.nDim, self.model.nDim))]
            ])

            # set up right-hand-side vector
            RHS = np.block([
                [np.array([self.prior.covarInvMean]).T],
                [np.array([sensorset.Lq[0].dot(measured)]).T],
                [np.zeros((self.model.nDim, 1))]
            ])

            # solve linear system
            v = la.solve(LHS, RHS)
            # TODO: For some reason, it takes way too long to use an iterative solver here. Probably because the
            # TODO: preconditioning is unclear.

        # identify the different parts of the solution (parameter, state, adjoint)
        v = np.split(v, [self.model.nPara, self.model.nPara + self.model.nDim, self.model.nPara + 2 * self.model.nDim])

        # compute difference in measurements
        y = sensorset.measure(v[1]).T

        return v[0], v[1], v[2], y

    def adjointsolve(self, sensorset, theta, sigma2, measured=None, state=None, **kwargs):
        """
        this function computes the solution of the adjoint equation in the MAP minimization problem for a linear
        forward model of type Au = Bm (u: state, m: parameter). In comparison to the adjoint equation that we get
        from the first order conditions in the minimization problem, the adjoint solution here is scaled by the
        factor sigma2. This is for numerical stability, as otherwise the norm of the adjoint equation blows up
        for decreasing sigma2 if the data cannot be obtained.

        This function is not in the forward model classes FEmodel or the RBmodel class, because adjoint equations are
        defined through the minimization problems that shall be solved. The forward model classes don't know about
        any minimization.

        If either no state or no measurements are given, the function only uses the other, given one.
        If neither is given, it uses all possible combinations of measurements (this corresponds to the case where
        measured it the nSensor x nSensor identity matrix)

        @param Lq: array of size (3,))
            Lq[0]: (inverse noise covariance operator) applied to L, shape (model.nDim, nSensors)
            Lq[1]: observation operator L, shape (nSensors, model.nDim)
            Lq[2]: L.T.dot(Lq[1]), shape (model.nDim, model.nDim)
        @param theta: hyper-parameter
        @param sigma2: scaling
        @param measured: measurements
        @param state: state vector, or basis of a subspace
        @param kwargs:
        @return:
        """
        Lq = sensorset.Lq

        # assembly of the model operator
        A_theta = self.model.assemble_lhs(self.model.Aq, theta)

        # distinguish between the different rhs cases
        if state is None:
            if measured is None:
                rhs = np.array([Lq[0]])
            else:
                rhs = Lq[0].dot(measured.T)
        else:
            if measured is None:
                rhs = - Lq[1].T.dot(Lq[0].T.dot(state))
            else:
                rhs = Lq[0].dot((measured - Lq[1].dot(state).T).T)
                # by definition: Lq[0]*Lq[1] = Lq[1]*Lq[0]
                # rhs = (Lq[0].dot(measured).T - Lq[1].T.dot(Lq[0].T.dot(state)).T).T
                # for ndarrays: if measured or state is in column format, this forces a row and then back to column
                # for arrays, the .T doesn't change anything.
                # old code with Lq[2]= Lq[1].T.dot(Lq[0].T): rhs = (Lq[0].dot(measured).T - Lq[2].dot(state).T).T

        # rhs = rhs / sigma2
        rhs = self.model.apply_BC_zeroDirichlet(rhs)

        # distinguish between sparse and dense solver routines
        if self.model.bool_sparseSolvers:

            # sparse iterative solve
            p, info = sla.cg(A_theta.T, rhs, tol=1e-12*sigma2, atol=1e-12)

            if info < 0:
                warnings.warn("Illegal input or breakdown in class_distribution_postGrandparents.adjointsolve")
            if info > 0:
                warnings.warn("convergence to tolerance not achieved, number of iterations, "
                              "in class_distribution_postGrandparents.adjointsolve")

        else:
            # direct solve
            p = la.solve(A_theta, rhs)

        return p
