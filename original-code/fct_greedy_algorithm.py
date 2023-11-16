# imports
import numpy as np

import time  # for timing computations

import warnings
warnings.filterwarnings(action="always")
# for warning, because raising them causes the program to stop.

import class_RBmodel  # class for RB model



def greedy_algorithm(fom, theta_iter, Xi_train, eps = 1e-02, nMax = 10, eps_expand = 1e-14, **kwargs):
    """
    Greedy Algorithm for the construction of an RB space:
    Iteratively expands RB space until maximum relative error bound is below the threshold eps

    :param fom: fullorder model (FE model)
    :param theta_iter: first hyper-parameter
    :param Xi_train: hyper-parameter training set
    :param eps: relative target accuracy
    :param nMax: maximum RB dimension
    :param eps_expand: threshold for considering vectors linearly independent
    :return:
    """

    tStart_greedy = time.time()
    """ 
    Initialization 
    """
    if kwargs.get("bool_freshStart", True):
        rom = greedy_initialization_freshStart(fom, theta_iter, Xi_train, tStart_greedy, eps, **kwargs)
    else:
        rom = greedy_initialization_fromMatrices(fom, Xi_train, eps, **kwargs)

    if kwargs.get("bool_print", False):
        print("RB initialization finished, duration: ", time.time() - tStart_greedy, " s.")

    # find next hyper-parameter with maximal relative error bound
    if kwargs.get("bool_randomSampling", False):
        theta_iter, para_iter, eps_iter, trainSize = find_next_random(rom, Xi_train, eps, **kwargs)
        rom.history["nTrain"] = [trainSize]
    else:
        theta_iter, para_iter, eps_iter = find_next(rom, Xi_train,
                                                    bool_substituteFE=kwargs.get("bool_substituteFE", False))

    """
    Greedy Algorithm Loop
    
    In each iteration, we find the hyperparameter hyper_iter in Xi_train for which the maximum relative error
    bound over the parameter space is the largest. We also find the parameter para_iter which maximizes the 
    relative error bound. For this combination, we then compute the FE solution and expand the RB space with it.
    """
    while eps_iter > eps and rom.nRB < nMax:

        # compute FE solution
        tStart_FE = time.time()
        FEsol = fom.forwardsolve(theta_iter, para_iter)
        if kwargs.get("bool_print", False):
            print("FE forwardsolve took ", time.time() - tStart_FE, " seconds.")

        # gather information about current approximation properties
        RBsol = rom.forwardsolve(theta_iter, para_iter)
        normRB = np.sqrt(rom.compute_norm2(RBsol))
        err = np.sqrt(rom.compute_difference(FEsol, RBsol))

        # expand RB space
        tStart_expand = time.time()
        bool_expanded, normFE, normFE_orig = rom.basis_expand(FEsol, eps=eps_expand, bool_expand=True)
        if kwargs.get("bool_print", False):
            print("RB expansion step, duration: ", time.time() - tStart_expand, " s.")

        # save gathered information
        rom.history["errBound"] = np.vstack((rom.history["errBound"], [eps_iter, err, normFE_orig, normFE, normRB]))
        rom.history["snapshots"] = np.vstack((rom.history["snapshots"], [theta_iter, para_iter]))

        if kwargs.get("bool_print", False):
            print("Values for RB Iteration: ", rom.nRB, eps_iter, err / normRB, eps_iter*normRB/err, "\n \n")
        else:
            print("Values for RB Iteration: ", rom.nRB, eps_iter, err / normRB, eps_iter*normRB/err)

        if not bool_expanded:
            # if the space was not expanded in the expansion step to avoid (near) linear dependence of the
            # basis vectors, then we are now stuck in a loop.
            warnings.warn("RB space was not expanded to avoid linear dependence of RB basis vectors. \
            Please set new threshold or re-think your problem and convergence threshold. We are nice \
            and return the current state for you.")
            rom.history["termination"] = "encountered loop"
            return rom

        # find next hyper-parameter with maximal relative error bound
        tStart_findNext = time.time()

        if kwargs.get("bool_randomSampling", False):
            theta_iter, para_iter, eps_iter, trainSize = find_next_random(rom, Xi_train, eps, **kwargs)
            rom.history["nTrain"] = np.vstack((rom.history["nTrain"], [trainSize]))
        else:
            theta_iter, para_iter, eps_iter = find_next(rom, Xi_train, bool_substituteFE=kwargs.get("bool_substituteFE", False))

        if kwargs.get("bool_print", False):
            print("FindNext took", time.time()-tStart_findNext, " seconds.")

    """
    Follow-Up
    
    For a last time, we gather some more information about the RB space
    """
    tStop_greedy = time.time()
    if kwargs.get("bool_print", False):
        print("Termination: Last values for RB Iteration: ", rom.nRB, eps_iter, "\n \n")
    else:
        print("Termination: Last values for RB Iteration: ", rom.nRB, eps_iter)

    # this is an unnecessary FE solve that we only do to have complete information. Hence we don't include it in
    # the timing process
    FEsol = fom.forwardsolve(theta_iter, para_iter)
    normFE = np.sqrt(fom.compute_norm2(FEsol))

    RBsol = rom.forwardsolve(theta_iter, para_iter)
    normRB = np.sqrt(rom.compute_norm2(RBsol))
    err = np.sqrt(rom.compute_difference(FEsol, RBsol))

    rom.history["errBound"] = np.vstack((rom.history["errBound"], [eps_iter, err, normFE, "unknown", normRB]))
    rom.history["greedy-time"] = tStop_greedy - tStart_greedy
    rom.history["snapshots"] = np.vstack((rom.history["snapshots"], [theta_iter, para_iter]))

    if eps_iter <= eps:
        rom.history["termination"] = "approximation threshold reached"
    else:
        rom.history["termination"] = "RB dimension threshold reached"

    return rom


def greedy_initialization_freshStart(fom, theta_iter, Xi_train, tStart_greedy, eps=1e-02, **kwargs):
    """
        Initialization

        we initialize with the solutions to all rhs. The reason we do this is so that we can guarantee that
        b( u , . ) neq 0 in YR' for all u in U,
        and that therefore the relative error bounds will be finite below. If b is parameter-dependent,
        this does not necessarily hold.
        """
    # TODO: find a better way that extends to more general and bigger spaces. This approach is infeasible if
    # TODO: the parameter dimension is higher or we cannot guarantee eta_inf > 0

    # bringing the RB model into existence
    rom = class_RBmodel.RBmodel(fom)

    if kwargs.get("bool_randomSampling", False):
        # save information about the random sampling process
        rom.history["type"] = "greedy, random samples"
        rom.history["training"] = kwargs.get("fct_Xi_train_description", "missing description")
        rom.history["nTrain_random_init"] = kwargs.get("nTrain_random_init", 100)
        rom.history["nTrain_counter_max"] = kwargs.get("nTrain_counter_max", 3)
    else:
        # save the information about the training set sampling process
        rom.history["type"] = "greedy, training set"
        rom.history["training"] = Xi_train

    rom.history["eps"] = eps
    rom.history["bool_substituteFE"] = kwargs.get("bool_substituteFE", False)

    rom.history["errBound"] = ["rel bound", "true err", "FE norm", "orthon. FE norm", "RB norm"]
    rom.history["snapshots"] = ["theta", "para"]

    # initialize space with all snapshots
    if rom._nPara == 1:

        FEsol = fom.forwardsolve(theta_iter, np.array([1.0]))
        rom.basis_initialize(FEsol)

    else:

        FEsol = fom.forwardsolve_all(theta_iter)
        rom.basis_initialize(FEsol[:, 0])

        for i in range(1, fom.nPara, 1):
            # gather information about the approximation qualities of the current RB space
            FEsol_temp = FEsol[:, i]
            normFE = np.sqrt(fom.compute_norm2(FEsol_temp))
            RBsol = rom.forwardsolve(theta_iter, np.eye(rom.nPara)[:, i])
            normRB = np.sqrt(rom.compute_norm2(RBsol))
            err = np.sqrt(rom.compute_difference(FEsol_temp, RBsol))
            rom.history["errBound"] = np.vstack((rom.history["errBound"], ["unknown", err, normFE, "unknown", normRB]))
            rom.history["snapshots"] = np.vstack((rom.history["snapshots"], [theta_iter, np.eye(rom.nPara)[:, i]]))

            # expand RB space
            rom.basis_expand(FEsol[:, i])

    return rom


def greedy_initialization_fromMatrices(fom, Xi_train, eps=1e-02, **kwargs):
    rom = class_RBmodel.RBmodel(fom)
    rom.feed_matrices(YR=kwargs["YR_feed"], **kwargs)

    if kwargs.get("bool_randomSampling", False):
        # save information about the random sampling process
        rom.history["type"] = "greedy, random samples"
        rom.history["training"] = kwargs.get("fct_Xi_train_description", "missing description")
        rom.history["nTrain_random_init"] = kwargs.get("nTrain_random_init", 100)
        rom.history["nTrain_counter_max"] = kwargs.get("nTrain_counter_max", 3)
    else:
        # save the information about the training set sampling process
        rom.history["type"] = "greedy, training set"
        rom.history["training"] = Xi_train

    rom.history["eps"] = eps
    rom.history["bool_substituteFE"] = kwargs.get("bool_substituteFE", False)

    rom.history["errBound"] = ["rel bound", "true err", "FE norm", "orthon. FE norm", "RB norm"]
    rom.history["snapshots"] = ["theta", "para"]

    return rom

def find_next(rom, Xi_train, bool_substituteFE = False):
    para_max = np.array([1.0])
    val_max = -1
    hyper_max = Xi_train[0, :]

    for i in range(Xi_train.shape[0]):

        errBound, para = rom.error_max(Xi_train[i, :], bool_substituteFE=bool_substituteFE)

        if errBound > val_max:
            hyper_max = Xi_train[i, :]
            val_max = errBound
            para_max = para

    if val_max < 0:
        warnings.warn("Search for next parameter failed to find an error bound >= 0.")

    return hyper_max, para_max, val_max


def find_next_random(rom, fct_Xi_train, eps, **kwargs):
    """
    Chooses the next hyper-parameter snapshot location based on a random sampling scheme

    @param rom:
    @param fct_Xi_train:
    @param kwargs:
    @return:
    """
    counter = 0
    nTrain = kwargs.get("nTrain_random_init", 100)
    nTrainTotal = 0;

    # initializations so that pycharm knows these are defined
    para_max = np.array([1.0])
    val_max = -1
    hyper_max = fct_Xi_train()

    while counter < kwargs.get("nTrain_counter_max", 3):

        for i in range(nTrain):

            theta = fct_Xi_train()
            errBound, para = rom.error_max(theta, bool_substituteFE=kwargs.get("bool_substituteFE", False))

            if errBound > val_max:
                hyper_max = theta
                val_max = errBound
                para_max = para

        nTrainTotal += nTrain

        if val_max < 0:
            warnings.warn("Search for next parameter failed to find an error bound >= 0.")

        if kwargs.get("bool_print", False):
            print("val_max = ", val_max, "with nTrain = ", nTrain, "samples.")

        if val_max > eps:
            return hyper_max, para_max, val_max, nTrainTotal
        else:
            counter = counter+1
            nTrain = 2*nTrain

    return hyper_max, para_max, val_max, nTrainTotal


