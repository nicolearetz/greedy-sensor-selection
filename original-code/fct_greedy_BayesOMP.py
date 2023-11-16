import scipy.linalg as la
# for dense linear algebra

import numpy as np
# for math functions

import warnings
# for giving out warnings (raising warnings interrupts the program)

import class_sensorSet


def compute_obsCoeff(SP_hyper, SP_noise):
    """
    computes teh observability coefficient for the sensors in the indexset on the state subspace Y_hyper

    We could catch the case where the parameter dimension is smaller than the number of sensors (in which case the
    observability coefficient is equal to zero), but since we need the corresponding eigenvector anyway, we don't
    do that.

    @param Y_hyper: state subspace
    @param SP_hyper: inner product on the subspace
    @param SP_noise: the inner product of the subspace's sensor measurements
    @return: observability coefficient and the parameter which achieves it
    """
    try:
        SP_hyper.shape[0]
    except IndexError:
        # catch the case in which the parameter-space is one dimensional
        return np.array([np.sqrt(SP_noise/SP_hyper)]), np.array([1.0/np.sqrt(SP_hyper)])

    eigval, eigvec = la.eigh(SP_noise, SP_hyper, eigvals=[0, 0])

    if SP_noise.shape[0] < SP_hyper.shape[0]:
        # less sensors than parameters -> observability coefficient equals zero
        eigval = np.array([0])

    return np.sqrt(np.abs(eigval[0])), eigvec

def select_sensors(fom, lib, theta, obsCoeff_stop, nSensors_stop, bool_paraRed = False, para = np.zeros([0,0]), sensorset = None, bool_prior = False, prior = None):
    """
    this function selects the sensors to iteratively increase the observability coefficient for one fixed hyper-
    parameter theta by using the fullorder model

    # TODO: implement termination criterion for stagnating improvement
    # TODO: implement option to instead use reduced order model, e.g. so that we can compare

    @param fom:
    @param lib:
    @param theta:
    @param obsCoeff_stop:
    @param nSensors_stop:
    @param bool_paraRed:
    @param para:
    @return:
    """

    # initialization
    if sensorset is None:
        sensorset = class_sensorSet.SensorSet(lib, nSensors_stop)
    obsCoeff_iter = np.zeros(nSensors_stop+1)

    # compute state subspace
    if bool_paraRed:
        # reduced parameter space case
        Y_hyper = fom.forwardsolve_paraRed(theta, para)

    else:
        # normal case
        Y_hyper = fom.forwardsolve_all(theta)

    if bool_prior:
        SP_hyper = prior.covar_inv
    else:
        SP_hyper = Y_hyper.T.dot(fom.SP.dot(Y_hyper))

    # compute start state
    if len(Y_hyper.shape) == 1:
        vecFE = Y_hyper
    else:
        vecFE = Y_hyper[:, 0]

    while sensorset.nSensors < nSensors_stop and obsCoeff_iter[sensorset.nSensors] < obsCoeff_stop:

        # compute measurements at current state
        measured = sensorset.measure(vecFE)

        # initialization for finding the maximum improvement
        improvement_max = -1
        j_max = 0
        w_max = None
        beta_max = None

        for j_iter in range(lib.nLibSensors):

            # compute possible improvement with sensor j_iter
            improvement, w_iter, beta_iter, j_iter_return = sensorset.compute_improvement(vecFE, measured, j_iter)

            # check if this is the maximum improvement
            if improvement > improvement_max:

                improvement_max = improvement
                w_max = w_iter
                beta_max = beta_iter
                j_max = j_iter_return

        # extend sensor set with sensor at index j_max
        bool_added = sensorset.add_sensor(j_max, w_max, beta_max)

        if not bool_added:
            # detects pathological behaviour, infinite loop
            warnings.warn("The sensorset couldn't be expanded. Sensor selection is terminated to avoid infinite loop. ")
            break

        # find minimum observability coefficient and associated state
        SP_noise = lib.compute_SPnoise(Y_hyper, sensorset.indexset)
        obsCoeff_iter[sensorset.nSensors], para = compute_obsCoeff(SP_hyper, SP_noise)
        vecFE = Y_hyper.dot(para)

        print("no. of sensors: ", sensorset.nSensors, ", min observability coefficient: ", obsCoeff_iter[sensorset.nSensors],
              ", last added sensor: ", j_max)

    sensorset.add_sensor_finished()
    obsCoeff_iter = obsCoeff_iter[range(sensorset.nSensors+1)]

    return sensorset, obsCoeff_iter

def select_sensors_hyperparameterized(rom, lib, Xi_train, obsCoeff_stop, nSensors_stop, sensorset = None, bool_prior = False, prior = None, **kwargs):
    """
    our sensor selection algorithm in the hyper-parameterized context. It works on an reduced order model that
    approximates the FE solution manifold over the parameter domain for each hyperparameter.

    TODO: only iterate over those hyper-parameters for which the threshold hasn't been reached yet
    TODO: implement termination criterion for stagnating improvement
    TODO: implement option to not compute all the solution manifolds
    TODO: implement having a random training set rather than a large fixed one

    @param rom: reduced order model
    @param lib: library from which to choose the sensors
    @param Xi_train: training set for the hyper-parameters
    @param obsCoeff_stop: stopping threshold for good observability coefficient
    @param nSensors_stop: stopping theshold, number of sensors
    @return: sensorset of chosen sensors, array with how the min observability coefficient changed
    """

    # initialization
    if sensorset is None:
        sensorset = class_sensorSet.SensorSet(lib, nSensors_stop)

    obsCoeff_iter = np.zeros((nSensors_stop + 1, 3))
    _nTrain = Xi_train.shape[0]

    # letting pycharm know that these will exist when they are first used
    w_max = None; beta_max = None
    SP_noise = None; SP_noise_factor = None; measured_RB = None; obs_coeff_min = np.infty
    j_min = None; para_min = None

    # compute the basis representation coefficients for the RB manifolds
    # this takes some memory, especially when Xi_train is big, but this reduces the cost of computing the next
    # hyper-parameter in each iteration from RB dimension to parameter dimension
    Mfd = np.zeros(_nTrain, dtype=object)
    SP_Mfd_diag = np.zeros(_nTrain, dtype=object)
    SP_Mfd_prior = np.zeros(_nTrain, dtype=object)  # this is new

    for i in range(_nTrain):
        Mfd[i] = rom.forwardsolve_all(Xi_train[i, :])
        SP_Mfd = Mfd[i].T.dot(rom.SP.dot(Mfd[i]))

        # identify if we should throw away some of the solutions
        eigvals, eigvecs = la.eigh(SP_Mfd)
        k = 0
        while eigvals[k] < 1e-06 and k < rom.nRB:
            k = k + 1

        # orthogonalize the basis functions
        Mfd[i] = Mfd[i].dot(eigvecs[range(k, rom.nPara, 1), :])
        SP_Mfd_diag[i] = eigvals[range(k, rom.nPara, 1)]
        SP_Mfd_prior[i] = eigvecs[range(k, rom.nPara, 1), :]  # this is new

    # compute first state of which to take the measurements
    vecRB = Mfd[0][:, SP_Mfd_diag[0].shape[0] - 1]
    vecFE = rom.compute_FErepresentation(vecRB)

    while sensorset.nSensors < nSensors_stop and obsCoeff_iter[sensorset.nSensors,0] < obsCoeff_stop:

        # compute measurements at current state
        measured = sensorset.measure(vecFE)

        # initialization for finding the maximum improvement
        improvement_max = -1
        j_max = 0

        for j_iter in range(lib.nLibSensors):

            # compute possible improvement with sensor j_iter
            improvement, w_iter, beta_iter, j_iter_return = sensorset.compute_improvement(vecFE, measured, j_iter)

            # check if this is the maximum improvement
            if improvement > improvement_max:

                improvement_max = improvement
                w_max = w_iter
                beta_max = beta_iter
                j_max = j_iter_return

        if improvement_max <= 0:
            warnings.warn("no improvement could be found, improvement_max = ", improvement_max)

        # extend sensor set with sensor at index j_max
        obsCoeff_iter[sensorset.nSensors, 2] = improvement_max
        bool_added = sensorset.add_sensor(j_max, w_max, beta_max)

        if not bool_added:
            # detects pathological behaviour, infinite loop
            warnings.warn("The sensorset couldn't be expanded. Sensor selection is terminated to avoid infinite loop. ")
            break

        # find minimum observability coefficient and associated state
        obs_coeff_min = np.infty
        if sensorset.nSensors == 1:
            SP_noise, SP_noise_factor, measured_RB = sensorset.compute_SPnoise(rom.YR)
        else:
            SP_noise, SP_noise_factor, measured_RB = sensorset.compute_SPnoise_update(rom.YR, SP_noise, SP_noise_factor, measured_RB)

        for j in range(_nTrain):

            # compute observability coefficient for hyper-parameter Xi_train[j, :]
            vectorRep = range(np.max([SP_Mfd_diag[j].shape[0]-sensorset.nSensors,0]), SP_Mfd_diag[j].shape[0])
            matTrans = Mfd[j][:, vectorRep]

            SP_noise_hyper = matTrans.T.dot(SP_noise.dot(matTrans))

            if (bool_prior):
                temp = SP_Mfd_prior[j][:, vectorRep]
                SP_hyper = temp.T.dot(prior.covar_inv.dot(temp)) # this is new
            else:
                SP_hyper = np.diag(SP_Mfd_diag[j][vectorRep])

            obs_coeff, para = compute_obsCoeff(SP_hyper, SP_noise_hyper)
            obsCoeff_iter[sensorset.nSensors, 1] = obsCoeff_iter[sensorset.nSensors, 1] + obs_coeff

            # check if new infimum has been found
            if obs_coeff < obs_coeff_min:
                obs_coeff_min = obs_coeff
                j_min = j
                para_min = para

        # compute FE state for which the observability coefficient is minimized
        obsCoeff_iter[sensorset.nSensors, 0] = obs_coeff_min
        obsCoeff_iter[sensorset.nSensors, 1] = obsCoeff_iter[sensorset.nSensors, 1]/_nTrain

        vectorRep = range(np.max([SP_Mfd_diag[j_min].shape[0]-sensorset.nSensors, 0]), SP_Mfd_diag[j_min].shape[0])
        matTrans = Mfd[j_min][:, vectorRep]
        vecRB = matTrans.dot(para_min)
        vecFE = rom.compute_FErepresentation(vecRB)

        if kwargs.get("bool_print", False):
            print("no. of sensors: ", sensorset.nSensors, ", min observability coefficient: ", obs_coeff_min, ", last added sensor: ", j_max)

    # finishing touches
    sensorset.add_sensor_finished()
    obsCoeff_iter = obsCoeff_iter[range(sensorset.nSensors+1), :]

    if kwargs.get("bool_print", False):
        print("Termination with ", sensorset.nSensors, " sensors at min observability coefficient: ", obs_coeff_min)

    return sensorset, obsCoeff_iter