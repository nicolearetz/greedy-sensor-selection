import numpy as np
# for general math functions

import scipy.linalg as la
# for linear algebra

import warnings
# for giving out warnings (raising warnings interrupts the program)

# We use scipy.linalg over numpy.linalg, because of the explanation at
# https://scipy.github.io/devdocs/tutorial/linalg.html

class RBmodel:

    def __init__(self, fom):
        """
        Initialization with the fullorder model.

        :param fom: fullorder model of FEmodel class
        """
        self.fom = fom

        # We don't have an RB space yet, but out of convention
        # we define all our class variables in the initialization
        self.YR = np.zeros([1, 1])
        self.SP = np.zeros([1, 1])
        self.SPm = fom.SPm

        self._nRB = 0
        self._nPara = fom.nPara
        self._nAq = fom.nAq
        self._nBq = fom.nBq

        self.Aq = np.empty([fom.nAq, 1], dtype=object)
        self.Bq = np.empty([fom.nBq, 1], dtype=object)

        self.matErrq = {
            "AA": np.empty([fom.nAq, fom.nAq], dtype=object),
            "AB": np.empty([fom.nAq, fom.nBq], dtype=object),
            "BB": np.empty([fom.nBq, fom.nBq], dtype=object)
        }

        self.history = {
            "type": "non-existent"
        }

        self.bool_sparseSolvers = False
        # this attribute tells other routines that use the model's matrices to use non-sparse solvers (i.e.
        # scipy.linalg)

    def feed_matrices(self, YR, **kwargs):
        """
        this function initalizes an RB model from given matrices

        @param YR: RB space, orthonormal w.r.t. fom.SP
        @param kwargs: dictionary for the option of feeding further matrices
        @return:
        """
        self.YR = YR
        self._nRB = self.YR.shape[1]
        self.SP = np.eye(self._nRB)
        self.history["type"] = "constructed from given matrices"

        if kwargs.get("bool_matrices2feed", False):
            self.Aq = kwargs["Aq"]
            self.Bq = kwargs["Bq"]
            self.matErrq = kwargs["matErrq"]
        else:
            self.basis_matrices(**kwargs)

    def forwardsolve(self, theta, para):
        """
        Forwardsolve of the form
        lhs(theta)*x = rhs(theta)*para
        in reduced dimension.
        Returns solution x as row vector.

        :param theta: hyper-parameter (possible nonlinear dependence)
        :param para: parameter (enters linearly)
        """
        A, B = self.fom.assemble_para(self.Aq, self.Bq, theta, para)
        return la.solve(A, B)

    def forwardsolve_all(self, theta):
        """
        Forwardsolve for all possible rhs's in the reduced system.
        Form of the equation:
        A(theta)*X = B(theta).
        Returns solution matrix X with X.shape = B.shape, unless B is a vector, then X is just an array of length n_FE.

        :param theta: hyper-parameter

        TODO: Test this method with more than one parameter dimension.
        """
        A, B = self.fom.assemble(self.Aq, self.Bq, theta)
        return la.solve(A, B)

    def assemble_lhs(self, Aq, theta):
        """
        Assembles and returns lhs(theta) = A(theta)

        :param Aq: array of lhs submatrices
        :param theta: hyper-parameter
        """
        return self.fom.assemble_lhs(Aq, theta)

    def assemble(self, Aq, Bq, theta):
        """
        Assembles and returns lhs(theta), rhs(theta) by calling the fullorder model function for it.

        :param qlhs: array of lhs submatrices
        :param qrhs: array of rhs submatrices
        :param theta: hyper-parameter
        """
        A = self.fom.assemble_lhs(Aq, theta)
        B = self.fom.assemble_rhs(Bq, theta)
        return A, B

    def apply_BC_zeroDirichlet(self, rhs):
        """
        we only need this function here for compatibility reasons with the fom (so that the posterior distribution
        doesn't need to know if it is treating a full- or reduced-order model). Here, this function does absolutely
        nothing.
        @param rhs:
        @return:
        """
        return rhs

    def compute_FErepresentation(self, coeffRB):
        """
        Computes the representation of the given RB coefficient vector in FE space
        :param coeffRB: vector of RB coefficients
        :return: representation in FE space
        """
        return self.YR.dot(coeffRB)

    def compute_difference(self, FEsol, coeffRB):
        """
        Computes the squared norm of the difference between a vector in FE space and the FE representation of a given
        RB basis coefficient vector

        :param solFE: vector in FE space
        :param coeffRB: coefficient vector
        :return: squared norm of the difference in FE space
        """

        diff = (FEsol - self.compute_FErepresentation(coeffRB).T).T
        # self.compute_FErepresentation(coeffRB) could be an array or a vector depending on the form of para

        return diff.T.dot(self.fom.SP.dot(diff))

    def compute_norm2(self, coeffRB):
        """
        Computes the squared norm of the FE representation of the coefficient vector.

        :param coeffRB: coefficient vector
        :return: squared norm of the vector in FE space
        """
        return (self.SP.dot(coeffRB)).T.dot(coeffRB)

    def error_bound(self, theta, para, coeff, bool_substitute=False):
        """
        Computes the a-posterior error bound at the RB solution coefficient vector so that
        solFE = self._fom.forwardsolve(theta, para)
        coeff = self.forwardsolve(theta, para)
        self.compute_difference(solFE, coeff) <= self.error_bound(theta, coeff)**2

        :param theta: hyper-parameter used for solving the RB forward problem
        :param coeff: RB solution vector, coeff = self.forwardsolve(theta, para)
        :return: error bound
        """
        if len(coeff.shape) == 1:
            coeff = np.array([coeff]).T

        if len(para.shape) == 1:
            para = np.array([para]).T

        AA, AB, BB = self.error_matrices(theta)
        res2 = coeff.T.dot(AA.dot(coeff))
        res2 = res2 + 2 * coeff.T.dot(AB.dot(para))
        res2 = res2 + para.T.dot(BB.dot(para))
        res2 = res2[0]

        if res2 < 1e-14:

            if bool_substitute:

                res2 = self.error_trueRes(theta, para, coeff)

            else:

                if res2 < -1e-12:
                    raise RuntimeError("the squared residual norm was far below 0: ||res||^2 = ", res2, "< -1e-12 < 0.")
                else:
                    # the squared residual norm can be smaller than zero due to round-off errors. As long as it is
                    # justifiable close to zero, we set it to zero. We include here values that are > 0 but still
                    # close to machine accuracy.
                    res2 = 0

        return np.sqrt(res2) / self.fom.coercivity_LB(theta)

    def error_matrices(self, theta):
        """
        Assembles the error matrices for the hyper-parameter theta

        :param theta: hyper-parameter
        :return: error matrices
            AA: shape [self._nr, self._nr]
            AB: shape [self._nr, self._nPara]
            BB: shape [self._nPara, self._nPara]
        """
        AA = self.fom.assemble_lhs(self.fom.assemble_lhs(self.matErrq["AA"], theta), theta)
        AB = self.fom.assemble_rhs(self.fom.assemble_lhs(self.matErrq["AB"], theta), theta)
        BB = self.fom.assemble_rhs(self.fom.assemble_rhs(self.matErrq["BB"], theta), theta)

        return AA, AB, BB

    def error_max(self, theta, bool_substituteFE=False):
        """
        Computes the largest relative a posteriori error bound that can be obtained over the parameter space:
        max(para) self.error_bound(theta, para, coeff) / ||YR(coeff)||
        where coeff = self.forwardsolve(theta, para)
        The maximum is obtained at the returned eigenvector.

        It can happen that the online, FE-dim-independent computation of the residual norm does not work due to
        round-off errors. In this case we compute it with the FE matrices. The values at which we decide to doe this
        are rather arbitrary.

        :param theta: hyperparameter
        :return:
            [0]: maximum error bound
            [1]: parameter for which maximum bound is obtained
        """
        # RB solutions for all parameters
        YR_hyper = self.forwardsolve_all(theta)
        SP_hyper = YR_hyper.T.dot(self.SP.dot(YR_hyper))
        # TODO: catch the case where the vectors in YR_hyper are linearly dependent and restrict to subspace

        # compute error matrix of shape [self.nPara, self.nPara]
        AA, AB, BB = self.error_matrices(theta)
        errMat = YR_hyper.T.dot(AA.dot(YR_hyper))
        errMat = errMat + YR_hyper.T.dot(AB) + AB.T.dot(YR_hyper)
        errMat = errMat + BB

        if len(errMat.shape) == 1:
            # if nRB == 1, then errMat is not in matrix format and then the eigenvalue problem runs into an error
            # hence we catch this case
            errMat = np.array([errMat])

        eigval, eigvec = la.eigh(errMat, b=SP_hyper, eigvals=(self._nPara - 1, self._nPara - 1))

        if eigval < 1e-14:

            if bool_substituteFE:
                print("the squared residual norm was far below 0: ||res||^2 = " + str(
                    eigval) + "< 1e-14. Recompute with FE matrices.")
                errMat = self.error_trueResMatrix(theta, YR_hyper)

                if len(errMat.shape) == 1:
                    # if nRB == 1, then errMat is not in matrix format and then the eigenvalue problem runs into an error
                    # hence we catch this case
                    errMat = np.array([errMat])

                eigval, eigvec = la.eigh(errMat, b=SP_hyper, eigvals=(self._nPara - 1, self._nPara - 1))
                print("not it's ", eigval)

            if np.abs(eigval) < 1e-13:
                # the squared residual norm can be smaller than zero due to round-off errors. As long as it is
                # justifiable close to zero, we set it to zero. We include here values that are > 0 but still
                # close to machine accuracy.
                eigval = np.array([0])

            if eigval < -1e-13:

                if bool_substituteFE:
                    warnings.warn("Even after computing the residual norm exactly with the FE matrices, it still has "
                                  " squared value " + str(eigval[0]) + "< -1e-13 < 0.")

                else:
                    warnings.warn("squared residual norm is negative with " + str(eigval[0]) + "< 0.")
                    print(theta)

                eigval = np.array([0])

        return np.sqrt(eigval[0]) / self.fom.coercivity_LB(theta), eigvec

    def error_trueRes(self, theta, para, coeff):
        """
        This function computes the residual norm directly with the FE matrices. This does take computations in
        FE dimension cost, but is more stable than the online computation.

        :param theta: hyper-parameter
        :param para: parameter
        :param coeff: RB solution coefficient vector
        :return: squared residual norm
        """
        A, B = self.fom.assemble_para(self.fom.Aq, self.fom.Bq, theta, para)
        res = B - A.dot(self.YR.dot(coeff))
        riesz = self.fom.compute_riesz(res)
        # riesz = sla.spsolve(self.fom.SP, res)
        return res.dot(riesz)

    def error_trueResMatrix(self, theta, YR_hyper):
        """
        This function computes the residual matrix for when we have not specified the parameters yet.

        :param theta: hyper-parameter
        :param YR_hyper: RB solution manifold
        :return: residual matrix
        """
        A, B = self.fom.assemble(self.fom.Aq, self.fom.Bq, theta)
        res = B - A.dot(self.YR.dot(YR_hyper))
        riesz = self.fom.compute_riesz(res)
        # riesz = sla.spsolve(self.fom.SP, res)

        return riesz.dot(res)

    def basis_initialize(self, FEsol):
        """
        This basis initializes the RB model with a one dimensional space, spanned by the input vector FEsol.
        This function is used for iterative basis expansions.

        :param FEsol: first basisvector of the RB space
        """
        # normalization
        norm2 = FEsol.T.dot(self.fom.SP.dot(FEsol))
        FEsol = FEsol / np.sqrt(norm2)
        self.YR = np.array([FEsol]).T
        self.SP = np.eye(1)
        self._nRB = 1

        # matrix generation
        self.basis_matrices()

        return True, np.sqrt(norm2), np.sqrt(norm2)

    def basis_expand(self, FEsol, eps=1e-12, bool_expand=True):
        """
        expands the current RB space with the vector FEsol

        matErrq["BB"] is independent of the RB space and is hence not expanded

        :rtype: bool
        :param FEsol: next basis vector to be included
        :param eps: threshold at which RB space is not expanded as vectors might become linearly dependent
        :param bool_expand: True if we shall expand the old matrices instead of computing all values anew
        :return: has the RB space been expanded?
        """
        if self.nRB == 0:
            # RB space is initialized instead of expanded
            return self.basis_initialize(FEsol)

        if len(FEsol.shape) == 2:
            # bring to right shape
            FEsol = FEsol.data

        # orthonormalization
        norm2_orig = FEsol.T.dot(self.fom.SP.dot(FEsol))
        FEsol = FEsol / np.sqrt(
            norm2_orig)  # we do this first normalization only so that our eps variable has a meaning

        for i in range(self.nRB):
            FEsol = FEsol - (FEsol.dot(self.fom.SP * self.YR[:, i])) * self.YR[:, i]

        norm2 = FEsol.dot(self.fom.SP.dot(FEsol))
        FEsol = FEsol / np.sqrt(norm2)

        if norm2 < eps:
            # we will not expand the RB space, since the vectors would become almost linearly dependent
            return False, np.sqrt(norm2), np.sqrt(norm2_orig)

        # expansion of the RB space
        self.YR = np.vstack([self.YR.T, FEsol]).T
        self._nRB = self._nRB + 1
        self.SP = np.eye(self._nRB)

        if self._nRB > 2 and bool_expand:

            # ideally we expand the matrices instead of computing every single entry anew.
            # However, for the special case where we go from dim YR = 1 to 2, we run into errors in the
            # computation of the error matrices because of the distinction between vectors and matrices.
            # Hence, in this special case we use the workaround of re-computing the previous values (all scalars)
            # The additional cost for this is far below what we need compute in the expansion anyway.

            # expansion of the matrices
            for i in range(self._nAq):

                A_temp = self.fom.Aq[i].dot(FEsol)
                riesz = self.fom.compute_riesz(A_temp)

                # expansion of Aq
                col = np.array([self.YR.T.dot(A_temp)]).T
                self.Aq[i] = np.hstack([self.Aq[i], col[0:(self._nRB - 1)]])
                self.Aq[i] = np.vstack([self.Aq[i], col.T])

                for j in range(self._nAq):
                    # expansion of matErrq["AA"]
                    col = np.array([riesz.T.dot(self.fom.Aq[j] * self.YR)]).T
                    self.matErrq["AA"][i][j] = np.hstack([self.matErrq["AA"][i][j], col[0:(self._nRB - 1)]])
                    self.matErrq["AA"][i][j] = np.vstack([self.matErrq["AA"][i][j], col.T])

                for j in range(self._nBq):
                    # expansion of matErrq["AB"]
                    self.matErrq["AB"][i][j] = np.vstack([self.matErrq["AB"][i][j], -riesz.T.dot(self.fom.Bq[j])])

            for i in range(self._nBq):
                # expansion of Bq
                self.Bq[i] = np.vstack([self.Bq[i], FEsol.T.dot(self.fom.Bq[i])])

        else:

            self.basis_matrices()
            # this recomputes all parts of the matrices instead of expanding the old ones

        return True, np.sqrt(norm2), np.sqrt(norm2_orig)

    def basis_matrices(self, **kwargs):
        """
        This functions computes the RB matrices for the RB space
        They are computed from scratch, not updated.
        """

        self.Aq = np.empty(self._nAq, dtype=object)
        self.Bq = np.empty(self._nBq, dtype=object)

        self.matErrq = {
            "AA": np.empty(self._nAq, dtype=object),
            "AB": np.empty(self._nAq, dtype=object),
            "BB": np.empty(self._nBq, dtype=object)
        }

        for i in range(self._nAq):

            if kwargs.get("bool_print", False):
                print("RBmodel.basis_matrices: constructing matrices (loop 1 of 2, i = ", i, " of ", self._nAq)

            self.Aq[i] = self.YR.T.dot(self.fom.Aq[i].dot(self.YR))

            A_temp = self.fom.Aq[i] * self.YR
            riesz = self.fom.compute_riesz(A_temp)

            self.matErrq["AA"][i] = np.empty(self._nAq, dtype=object)
            self.matErrq["AB"][i] = np.empty(self._nBq, dtype=object)

            for j in range(self._nAq):

                self.matErrq["AA"][i][j] = riesz.T.dot(self.fom.Aq[j] * self.YR)

                if kwargs.get("bool_print", False):
                    print("RBmodel.basis_matrices: constructing matrices (loop 1 of 2, : i = ", i, ", j = ", j, " of ", self._nAq)

            for j in range(self._nBq):

                self.matErrq["AB"][i][j] = -self.fom.Bq[j].T.dot(riesz).T

                if kwargs.get("bool_print", False):
                    print("RBmodel.basis_matrices: constructing matrices (loop 1 of 2, : i = ", i, ", j = ", j, " of ", self._nBq)

        for i in range(self._nBq):

            if kwargs.get("bool_print", False):
                print("RBmodel.basis_matrices: constructing matrices (loop 2 of 2, i = ", i, "of ", self._nBq)

            # if Bq is sparse and YR dense, then YR.T.dot(Bq[0]) is not the matrix multiplication
            self.Bq[i] = self.fom.Bq[i].T.dot(self.YR).T

            riesz = self.fom.compute_riesz(self.fom.Bq[i])

            self.matErrq["BB"][i] = np.empty(self._nBq, dtype=object)

            for j in range(self._nBq):

                self.matErrq["BB"][i][j] = self.fom.Bq[j].T.dot(riesz).T
                # matErrq["BB"] is independent of the RB space

                if kwargs.get("bool_print", False):
                    print("RBmodel.basis_matrices: constructing matrices (loop 2 of 2, : i = ", i, ", j = ", j, " of ", self._nBq)

    @property
    def nRB(self):
        """getter for _nRB (RB dimension)"""
        return self._nRB

    @property
    def nPara(self):
        """getter for _nPara (Parameter dimension)"""
        return self._nPara

    @property
    def nDim(self):
        """returns the forward dimension"""
        return self._nRB

    @property
    def nAq(self):
        """
        the underscore in _nBq means, it's supposed to be a protected variable. I didn't know that when
        I named it that way...
        """
        return self._nAq

    @property
    def nBq(self):
        """
        the underscore in _nBq means, it's supposed to be a protected variable. I didn't know that when
        I named it that way...
        """
        return self._nBq
