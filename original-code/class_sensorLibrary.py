import numpy as np
# for general maths functions

# for dense linear algebra
import scipy.linalg as la


class SensorLibrary:

    def __init__(self, arguments):
        """
        Initialization
        """
        self.nLibSensors = arguments["L"]
        self.centers = np.zeros([self.nLibSensors, 2])

        self.customization(arguments)

    def customization(self, arguments):
        raise NotImplementedError("This function needs to be implemented in the subclass")

    def sensor_measure_i(self, vecFE, index):
        raise NotImplementedError("This function needs to be implemented in the subclass")

    def sensor_measure_I(self, vecFE, indexset):
        """
        Takes the measurements of the state vecFE for all sensors with index in the indexset
        @param vecFE: state to be measured
        @param indexset: indices of the sensors
        @return: measurements of sensors in array with shape (len(indexset),)
        """
        if len(vecFE.shape) == 1:
            # we are indeed measuring a vector

            measured = np.zeros(len(indexset), 1)
            for i in range(len(indexset)):
                measured[i, 0] = self.sensor_measure_i(vecFE, indexset[i])

        else:
            # we want to measure the columns of a matrix individually

            nVectors = vecFE.shape[1]
            measured = np.zeros(len(indexset), nVectors)

            for j in range(nVectors):
                for i in range(len(indexset)):
                    measured[i, j] = self.sensor_measure_i(vecFE, indexset[i])

        return measured

    def sensor_measure_all(self, vecFE):
        """
        This function computes the measurements of all sensors in the library

        :param vecFE: state of which to take measurements
        :return: vector with measurements
        """
        return self.sensor_measure_I(vecFE, range(self.nLibSensors))

    def sensor_covariance_ij(self, index_i, index_j):
        """
        This function returns the covariance between the sensors with indices i and j

        :param index_i:
        :param index_j:
        :return: covariance between the sensors
        """
        raise NotImplementedError("This function needs to be implemented in the subclass")

    def sensor_covariance_Ij(self, indexset, index_j):
        """
        This function returns the covariance vector between the sensors in the indexset and the sensor with index j
        The variance of the sensor at index j is not included in this list

        @return: array of shape (len(indexset),)
        @param indexset:
        @param index_j:
        """
        nIndexset = len(indexset)
        covariance = np.zeros([nIndexset])

        for i in range(nIndexset):
            covariance[i] = self.sensor_covariance_ij(i, index_j)

        return covariance

    def sensor_covariance_I(self, indexset):
        """
        returns the covariance matrix for all the sensors in the indexset

        @param indexset:
        @return: array of shape (len(indexset), len(indexset))
        """
        nIndexset = len(indexset)
        covariance = np.zeros([nIndexset, nIndexset])

        for i in range(nIndexset):

            covariance[i, i] = self.sensor_covariance_ij(i, i)

            for j in range(i):

                covariance[i, j] = self.sensor_covariance_ij(i, j)
                covariance[j, i] = covariance[i, j]

        return covariance

    def sensor_covariance_all(self):
        """
        returns the covariance matrix for all sensors in the library
        :return:
        """
        return self.sensor_covariance_I(range(self.nLibSensors))

    def sensor_covarianceInv_apply(self, measured, indexset):
        """
        applies the inverse of the covariance matrix (that is associated to the indexset) to the measurements

        @param measured: measurements taken with sensors in the indexset
        @param indexset: sensors considered
        @return: (cov(I).inv * measured)
        """
        covMat = self.sensor_covariance_I(indexset)
        return la.solve(covMat, measured)

    def compute_SPnoise(self, Y_hyper, indexset):
        """
        computes the noise inner product on a given state subspace Y_hyper

        @param Y_hyper:
        @param indexset:
        @return:
        """
        measured = self.sensor_measure_I(Y_hyper, indexset)
        SP_noise = self.sensor_covarianceInv_apply(measured, indexset)
        return measured.T.dot(SP_noise)

    def compute_noisenorm2(self, measured, indexset):
        """
        computes the squared noise-norm of the measurements

        @param measured: measurements
        @param indexset: the indices of the chosen sensors
        @return:
        """
        return measured.T.dot(self.sensor_covarianceInv_apply(measured, indexset))[0][0]

    def sensor_positions(self, indexset):
        """
        gives the centers associated to the sensors in the indexset
        @param indexset:
        @return: array of shape (len(indexset), 3)
        """
        return self.centers[indexset, :]

    def compute_L(self, indexset, subspace=None):
        """
        This function computes the a matrix L = L(indexset) such that
        L.dot(vecCoeff) = self.sensor_measure_I(subspace.dot(vecCoeff), indexset),
        for all vecCoeff in R^nDim - if subspace is given, otherwise
        L.dot(vecFE) = self.sensor_measure_I(vecFE, indexset)
        for all FE vectors (in this case we need more information and the function needs to be implemented in the
        subclass)

        @param indexset: indices at which to take measurements
        @param subspace: matrix of shape (nFE, nDim)
        @return:
        """
        if subspace is None:
            raise NotImplementedError("This part needs to be implemented in a subclass")
        else:
            return self.sensor_measure_I(subspace, indexset)
