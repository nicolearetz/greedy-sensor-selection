import numpy as np
# for general maths functions

import scipy.linalg as la
# for dense linear algebra

import warnings
# for giving out warnings (raising warnings interrupts the program)


class SensorSet:

    def __init__(self, lib, nSensorsMax):
        """
        initialization of sensorSet class. It knows it's library lib, otherwise it doesn't know it's sensors yet.
        The library will be used for calling the action of the sensors.

        @param lib: library from which the sensors will come
        """
        self.lib = lib
        self.indexset = []
        self._nSensors = 0
        self._nSensorsMax = nSensorsMax
        self.covar = np.zeros([nSensorsMax, nSensorsMax])
        self.centersMax = np.zeros([nSensorsMax, lib.centers.shape[1]])
        self.LCholMax = np.zeros([nSensorsMax, nSensorsMax])
        self.bool_finished = False
        self._Lq = None
        self._Lq2 = None

    def add_sensor_set(self, new_indexset):
        """
        adds all sensors from a list of indices to the sensorSet

        @param new_indexset: list of indices
        """
        for i in new_indexset:
            self.add_sensor(i)

    def add_sensor(self, index, w=None, beta=None):
        """
        expands the sensor set with the sensor at index in the library.

        @param index: index of new sensor
        @param w: new line of the Cholesky decomposition
        @param beta: new diagonal entry of the Cholesky decomposition
        @return: True if sensor was added, False + warning otherwise
        """
        if self.bool_finished:
            warnings.warn("Sensor selection was already finished, sensor set is not expanded")
            return False

        if self._nSensors == 0:
            self.add_sensor_first(index)
            return True

        if index in self.indexset:
            return False

        if self._nSensors + 1 > self._nSensorsMax:
            warnings.warn("maximum number of sensors reached, sensor set is not expanded")
            return False

        # get entries for the covariance matrix
        vec_covar = self.lib.sensor_covariance_Ij(self.indexset, index)
        var = self.lib.sensor_covariance_ij(index, index)
        self.covar[range(self._nSensors), self._nSensors] = vec_covar.T
        self.covar[self._nSensors, range(self._nSensors)] = vec_covar
        self.covar[self._nSensors, self._nSensors] = var

        # expand Cholesky decomposition
        if w is None:
            w = la.solve_triangular(self.LChol, vec_covar, lower=True)
            beta = np.sqrt(var - w.T.dot(w))

        self.LCholMax[self._nSensors, range(self._nSensors)] = w
        self.LCholMax[self._nSensors, self._nSensors] = beta

        # remember the center of the new sensor
        self.centersMax[self._nSensors, :] = self.lib.sensor_positions(index)

        # update indexset
        self.indexset.append(index)
        self._nSensors = self._nSensors + 1

        return True

    def add_sensor_first(self, index):
        """
        Adds the first sensor to the sensor set

        @param index: index of the first sensor
        @return:
        """
        self.indexset.append(index)
        self._nSensors = 1
        self.covar[0, 0] = self.lib.sensor_covariance_ij(index, index)
        self.centersMax[0, :] = self.lib.sensor_positions(index)
        self.LCholMax[0, 0] = np.sqrt(self.covar[0, 0])

    def add_sensor_finished(self):
        """
        finished the sensor selection, sensor set can no longer be expanded
        """
        self.centersMax = self.centersMax[range(self._nSensors), :]
        self.LCholMax = self.LCholMax[np.ix_(range(self._nSensors), range(self._nSensors))]
        self.covar = self.covar[np.ix_(range(self._nSensors), range(self._nSensors))]
        self._nSensorsMax = self._nSensors
        self.bool_finished = True

    def measure(self, vecFE):
        """
        measures with the sensors in the sensorset

        @param vecFE: state to take the measurements of
        @return: measurements of sensors in array with shape (len(indexset),)
        """
        return self.lib.sensor_measure_I(vecFE, self.indexset)

    def compute_SPnoise(self, Y_sub):
        """
        Computes the noise inner product matrix, a rank factorization of it, and the measurements on the given state subspace

        The factor matrix A := SP_noise_factor is a factorization of the form
        SP_noise = A.T * A,
        but A doesn't have any special form. It has shape (self.nSensors, dim(Y_sub))

        SP_noise is not necessarily positive definite - only once enough sensors have been added so that the
        observability coefficient is >0

        %TODO: maybe rename SP_noise

        @param Y_sub:
        @return:
        """

        # compute measurements
        measured = self.measure(Y_sub)

        # compute factor matrix with the cholesky factorization from the sensor covariance matrix
        SP_noise_factor = la.solve_triangular(self.LChol, measured, lower=True)

        return SP_noise_factor.T.dot(SP_noise_factor), SP_noise_factor, measured

    def compute_SPnoise_update(self, Y_sub, SP_noise_old, SP_noise_factor_old, measured_old):
        """
        Updates the noise inner product matrix, a rank factorization of it, and the measurements on a given state
        subspace after sensors have been added.

        The factor matrix A := SP_noise_factor is a factorization of the form
        SP_noise = A.T * A,
        but A doesn't have any special form. It has shape (self.nSensors, dim(Y_sub))

        SP_noise is not necessarily positive definite - only once enough sensors have been added so that the
        observability coefficient is >0

        @param Y_sub: state subspace
        @param SP_noise_old: noise inner product matrix for old sensor set
        @param SP_noise_factor_old: matrix factorization of old noise inner product matrix
        @param measured_old: set of measurements taken with the old sensor set
        @return: noise inner product matrix, its matrix square root, measurements for space Y_sub
        """
        nSensors_old = SP_noise_factor_old.shape[0]

        # compute new measurements only
        measured_new = self.lib.sensor_measure_I(Y_sub, self.indexset[slice(nSensors_old, self.nSensors)])
        measured = np.vstack([measured_old, measured_new])

        # compute new part of the factor matrix
        wx_temp = self.LChol[range(nSensors_old, self.nSensors), range(nSensors_old)].dot(SP_noise_factor_old)
        SP_noise_factor_new = la.solve_triangular(self.LChol[np.ix_(range(nSensors_old, self.nSensors), range(nSensors_old, self.nSensors))], measured_new - wx_temp, lower=True)
        SP_noise_factor = np.vstack([SP_noise_factor_old, SP_noise_factor_new])

        # compute noise inner product matrix
        SP_noise = SP_noise_old + SP_noise_factor_new.T.dot(SP_noise_factor_new)

        return SP_noise, SP_noise_factor, measured

    def compute_improvement(self, vecFE, measured, index_j):
        """
        This function computes how much the observability of the vector vecFE would improve by adding the index_j to
        the indexset.

        #improvement, w_iter, beta_iter = compute_improvement(sensorset.LChol, vecFE, measured, lib, sensorset.indexset, j_iter)

        @param L_chol: cholesky factorization of current noise inner product (sensors in indexset)
        @param vecFE: state, for which the observability shall be improved
        @param measured: measurements of the state with the sensors so far
        @param lib: library from which the new sensor is taken
        @param indexset: indices of current sensor
        @param index_j: proposed new sensor index
        @return: expected improvement and additional entries for new cholesky factorization
        """
        # compute the new measurement at sensor index_j
        measured_new = self.lib.sensor_measure_i(vecFE, index_j)

        if len(measured_new.shape) == 1:
            measured_new = np.array([measured_new])

        if len(self.indexset) == 0:
            # catch the case where we haven't selected any measurements yet

            var = self.lib.sensor_covariance_ij(index_j, index_j)
            improvement = measured_new ** 2 / var
            return improvement[0], np.array([]), np.sqrt(var), index_j

        # get information about the covariance
        vec_cov = self.lib.sensor_covariance_Ij(self.indexset, index_j)
        var = self.lib.sensor_covariance_ij(index_j, index_j)

        # compute new entries of the inverse of the new Cholesky factorization
        w = la.solve_triangular(self.LChol, vec_cov, lower=True)
        beta = var - w.T.dot(w)
        # since the noise covariance matrix is positive definite, the cholesky decomposition exists and beta > 0 for
        # index_j not in indexset.
        # if index_j is already in the indexset, then beta = 0

        if beta < 1e-14:
            # check for singularity of the covariance matrix
            return -1, w, beta, index_j
        else:
            beta = np.sqrt(beta)

        x = -la.solve_triangular(self.LChol.T, w, lower=False) / beta

        # compute improvement
        improvement = ((x.T.dot(measured) + measured_new / beta) ** 2)[0][0]

        return improvement, w, beta, index_j

    def compute_Lq(self, subspace=None):
        """
        This function computes the observability operator, applies the inverse noise covariance matrix to it,
        and then its adjoint. All three results are saved.

        @param subspace: if we only want to consider a subspace in the future, this is basically an option for the
            RBsensorSet class

        @return: Lq: array of size (2,))
            Lq[0]: (inverse noise covariance operator) applied to L, then all transposed, shape (model.nDim, nSensors)
            Lq[1]: observation operator L, shape (nSensors, model.nDim)

        @note:
            Lq used to have an additional entry
            Lq[2] = Lq[1].T.dot(Lq[0].T), shape (model.nDim, model.nDim)
            but it should be cheaper to compute its action via the formula than computing it first and applying a
            whole nDim x nDim matrix.
        """
        if subspace is None and self._Lq is not None:
            return self._Lq

        #Lq = np.zeros(3, dtype=object)
        Lq = np.zeros(2, dtype=object)
        Lq[1] = self.lib.compute_L(self.indexset, subspace)

        # apply inverse noise covariance matrix
        Lq[0] = la.solve_triangular(self.LChol, Lq[1].todense(), lower=True)
        Lq[0] = la.solve_triangular(self.LChol, Lq[0], lower=True, trans="T")
        Lq[0] = Lq[0].T

        # apply adjoint of observation operator
        #Lq[2] = Lq[1].T.dot(Lq[0].T)

        if subspace is None and self.bool_finished:
            self._Lq = Lq

        return Lq

    def compute_norm2(self, measured):
        """
        This function computes the squared norm of the measurements in the noise norm induced by the (inverse of the)
        noise covariance matrix.

        @param measured:
        @return:
        """
        temp = la.solve_triangular(self.LChol, measured, lower=True)
        return temp.T.dot(temp)

    @property
    def LChol(self):
        if self.bool_finished:
            return self.LCholMax
        else:
            return self.LCholMax[np.ix_(range(self._nSensors), range(self._nSensors))]

    @property
    def centers(self):
        if self.bool_finished:
            return self.centersMax
        else:
            return self.centersMax[range(self._nSensors), :]

    @property
    def nSensors(self):
        return self._nSensors

    @property
    def Lq(self):
        if self._Lq is None:
            return self.compute_Lq()
        else:
            return self._Lq

    @property
    def Lq2(self):
        if self._Lq2 is None:
            self._Lq2 = self.Lq[1].T.dot(self.Lq[0].T)
        return self._Lq2