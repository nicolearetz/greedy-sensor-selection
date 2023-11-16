import scipy as sp
# for sparse matrices

import scipy.sparse.linalg as sla
# for linear algebra with sparse matrices

import warnings
warnings.filterwarnings(action="always")
# for giving out warnings (raising warnings interrupts the program)

import numpy as np


# for general math functions

class FEmodel:
    """
    general FE model class for linear problems of the form
    A(theta)*x = B(theta)*para,
    where
    theta is a hyper-parameter (enters non-linearly), para is a linear parameter, A and B are the lhs and rhs system
    matrices that are assembled from submatrices from an affine decomposition
    """

    def __init__(self, Aq, Bq, SP, SPm=None, zeroDirichlet=None):
        """
        Initialization of the class with FE matrices.

        @param Aq: list of FE submatrices of length len(Aq) for assembling the lhs
        @param Bq: list of FE submatrices of length len(Bq) for assembling the rhs
        @param SP: inner product matrix on FE space
        @param SPU: innere product matrix on parameter space, initialized as identity if not given
        @param zeroDirichlet: indizes that have to be set to zero for zero-Dirichlet boundary conditions, also see
            description in function `apply_BC_zeroDirichlet`
        """
        self.Aq = Aq
        self.Bq = Bq
        self.SP = SP

        # save FE and parameter dimension
        [self.nFE, self.nPara] = Bq[0].shape
        self.nAq = len(Aq)
        self.nBq = len(Bq)

        if SPm is None:
            self.SPm = np.eye(self.nPara)
            print("In FEmodel initialization: No inner product was defined on the parameter space, assuming orthonormality.")
        else:
            self.SPm = SPm

        self.bool_sparseSolvers = True
        # this attribute tells other routines that use the model's matrices to use sparse solvers (i.e.
        # scipy.sparse.linalg)

        # nodes with zero-Dirichlet boundary condition
        # for explanation / reason behind this, see description of function `apply_BC_zeroDirichlet`
        self.zeroDirichlet = zeroDirichlet

    def forwardsolve(self, theta, para):
        """
        Forwardsolve of the form
        lhs(theta)*x = rhs(theta)*para.
        Returns solution x as row vector.

        :param theta: hyper-parameter (possible nonlinear dependence)
        :param para: parameter (enters linearly)
        """
        A, B = self.assemble_para(self.Aq, self.Bq, theta, para)

        # slow direct solve
        # x = sla.spsolve(A, B)

        x, info = sla.cg(A, B, tol=1e-12, atol=1e-12)
        if info < 0:
            warnings.warn("Illegal input or breakdown in FEmodel.forwardsolve")
        if info > 0:
            warnings.warn("convergence to tolerance not achieved, number of iterations, in FEmodel.forwardsolve")

        return x

    def forwardsolve_paraRed(self, theta, paraMat):
        """
        Forwardsolve for all possible rhs's with parameter in paraMat.

        @param theta:
        @param paraMat:
        @return:
        """
        raise NotImplementedError("Reduction in parameter space still needs to be implemented")

    def forwardsolve_all(self, theta):
        """
        Forwardsolve for all possible rhs's.
        Form of the equation:
        A(theta)*X = B(theta).
        Returns solution matrix X with X.shape = B.shape, unless B is a vector, then X is just an array of length n_FE.

        :param theta: hyper-parameter
        """
        A, B = self.assemble(self.Aq, self.Bq, theta)

        # slow direct solve
        # x = np.zeros((self.nPara, self.nFE))
        # for i in range(self.nPara):
        #     x[i, :] = sla.spsolve(A, B[:, i].todense())
        # x = x.T
        # we could do this a bit better by just using x = sla.spsolve(A, B), but then we run into problems with the
        # dimensions and orientations of the solution vectors.

        x = np.zeros((self.nPara, self.nFE))
        for i in range(self.nPara):
            x[i, :], info = sla.cg(A, B[:, i], tol=1e-12, atol=1e-12)
            if info < 0:
                warnings.warn("Illegal input or breakdown in FEmodel.forwardsolve_all")
            if info > 0:
                warnings.warn(
                    "convergence to tolerance not achieved (number of iterations exceeded), i = " + str(i) + ", in FEmodel.forwardsolve_all")
        x = x.T

        return x

    def assemble(self, Aq, Bq, theta):
        """
        Assembles and returns lhs(theta), rhs(theta)

        :param qlhs: array of lhs submatrices
        :param qrhs: array of rhs submatrices
        :param theta: hyper-parameter

        Note:
        We could implement this routine without the arguments Aq and Bq by just calling the class variables, but
        we want to be able to access the assembly routine with our RB matrices from outside.

        """
        A = self.assemble_lhs(Aq, theta)
        B = self.assemble_rhs(Bq, theta)
        return A, B

    def assemble_lhs(self, Aq, theta):
        """
        Assembles and returns lhs(theta) = A(theta)

        :param Aq: array of lhs submatrices
        :param theta: hyper-parameter

        Note:
        We could implement this routine without the arguments Aq and Bq by just calling the class variables, but
        we want to be able to access the assembly routine with our RB matrices from outside.
        """
        raise NotImplementedError("The function needs to be implemented in the subclass.")

    def assemble_rhs(self, Bq, theta):
        """
        Assembles and returns rhs(theta) = B(theta)

        :param Bq: array of rhs submatrices
        :param theta: hyper-parameter

        Note:
        We could implement this routine without the arguments Aq and Bq by just calling the class variables, but
        we want to be able to access the assembly routine with our RB matrices from outside.
        """
        raise NotImplementedError("The function needs to be implemented in the subclass.")

    def assemble_para(self, Aq, Bq, theta, para):
        """
        Assembles and returns lhs(theta), rhs(theta)*para

        :param qlhs: array of lhs submatrices
        :param qrhs: array of rhs submatrices
        :param theta: hyper-parameter
        :param para: parameter

        Note:
        We could implement this routine without the arguments Aq and Bq by just calling the class variables, but
        we want to be able to access the assembly routine with our RB matrices from outside.
        """
        raise NotImplementedError("The function needs to be implemented in the subclass.")

    def apply_BC_zeroDirichlet(self, rhs):
        """
        This function applies zero-Dirichlet (zeroD) boundary conditions (BC) to a rhs vector for solving the problem
        A(theta) u = rhs
        or
        A(theta).T u = rhs.

        Ideally, the FE space in which we are operating has only basis functions which already comply with
        zero-Dirichlet boundary conditions. However, some FE libraries, such as MOOSE, include basis vectors for the
        boundary conditions for morevgenerality, and the linear system for solving the PDE then includes equations for
        setting their coefficientsvto zero (or whichever boundary condition is supposed to hold for the forward
        problem). This is what this function does.

        As far as the matrices are concerned: as soon as we come into a position where we want to solve the adjoint
        problem, we need to make sure the matrices are constructed in a way that the adjoint operator A.T

        The reason we are only considering zero-Dirichlet boundary conditions here (as opposed to non-Dirichlet) is
        because we are solving the equation
        find u in U s.t. a(u,v) = b(m, v) for all v in U (or v in V once we switch over to Petrov-Galerkin)
        where U is a Hilbert space. Since Dirichlet boundary conditions are properties of the set we search in,
        imposing non-zero Dirichlet boundary conditions makes that U is an affine, but not a linear space any more.
        We can, however, get to the zero-Dirichlet case via an affine transformation. So we can assume zero-Dirichlet
        boundary conditions without loss of generality.

        @todo: think about and test if this works as easily when with FE spaces with higher polynomial order than
        @todo: piecewise linear, and in higher dimensions.
        @note: It should, because at higher polynomial order, if we only force the values at the nodes to be zero of
        the representation as function, then between the boundary nodes the function is still != zero. This argument
        only hold though if the polynomial order is different along the boundary for basis functions centered at the
        same boundary node.

        @param rhs:
        @return:
        """
        if self.zeroDirichlet is not None:
            for i in self.zeroDirichlet:
                # set the
                rhs[i] = 0

        return rhs

    def coercivity_LB(self, theta):
        """
        Returns the coercivity lower bound for A(theta)

        :param theta: hyperparameter
        """
        raise NotImplementedError("The function needs to be implemented in the subclass.")

    def continuity_UB(self, theta):
        """
        Returns the coercivity lower bound for A(theta)

        :param theta: hyperparameter
        """
        raise NotImplementedError("The function needs to be implemented in the subclass.")

    def compute_norm2(self, vec):
        """
        Computes the squared norm of the vector input

        :param vec: FE vector of which to compute the norm
        :return: squared norm
        """
        if len(vec.shape) == 2 and vec.shape[1] == self.nFE and vec.shape[0] != self.nFE:
            # catch the case in which we are given a row matrix
            vec = vec.T

        return vec.T.dot(self.SP.dot(vec))

    def compute_riesz(self, fct):
        """
        computes the Riesz representation

        @note:
        we are computing the Riesz representation in the space with zero-Dirichlet boundary conditions. Hence, for
        the FE basis functions corresponding to the indices on the zero-Dirichlet boundary condition the fct always
        returns zero. Hence, we can, without loss of generality, assume that the vector-representaton of the fct
        just has a zero for this entry.

        :param fct:
        :return:
        """

        if fct.shape == (self.nFE, 1) or fct.shape == (self.nFE,):

            fct_zD = self.apply_BC_zeroDirichlet(fct)
            x, info = sla.cg(self.SP, fct_zD, tol=1e-12, atol=1e-12)

            if info < 0:
                warnings.warn("Illegal input or breakdown in FEmodel.forwardsolve")
            if info > 0:
                warnings.warn("convergence to tolerance not achieved, number of iterations, in FEmodel.forwardsolve")

        else:

            x = np.zeros((fct.shape[1], self.nFE))
            for i in range(fct.shape[1]):

                fct_zD = self.apply_BC_zeroDirichlet(fct[:, i])
                x[i, :], info = sla.cg(self.SP, fct_zD, tol=1e-12, atol=1e-12)

                if info < 0:
                    warnings.warn("Illegal input or breakdown in FEmodel.forwardsolve")
                if info > 0:
                    warnings.warn(
                        "convergence to tolerance not achieved, number of iterations, in FEmodel.forwardsolve")
            x = x.T

        return x

    @property
    def nDim(self):
        """returns the forward dimension"""
        return self.nFE
