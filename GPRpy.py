import numpy as np
from matplotlib import pyplot as plt
import warnings
from scipy.optimize import minimize as fmin
from scipy.optimize import brute as brute_optim
#%%


class GPRpy(object):

    # =============================================================================
    #     GPR: GAUSSIAN PROCESS REGRESSOR
    #     main class
    #
    #     by Filippo Maria Castelli
    #     last major update 22/02/2019
    # =============================================================================

    # =============================================================================
    #     INIT
    # =============================================================================
    def __init__(
        self, x, y, x_guess, kernel=False, kernel_params=False, R=0, normalize_y=True
    ):

        self.x = np.squeeze(x)
        self.y_input = np.squeeze(y)
        self.x_guess = np.squeeze(x_guess)
        self.N = len(self.x)
        self.R = np.squeeze(R)

        self.kernel = kernel if kernel else self.gaussian_kernel

        if kernel_params == {}:
            warnings.warn(
                "params for {} kernel not set!, remember to set them before using the predict method".format(
                    kernel
                )
            )
        else:
            self.update_params(kernel_params)

        self.kernel_arguments = self.find_arguments(self.kernel)

        self.normalize_y = normalize_y
        if self.normalize_y == True:
            self.y_mean = np.mean(self.y_input)
            self.y = self.y_input - self.y_mean
        else:
            self.y = self.y_input

    # =============================================================================
    #  KERNELS:
    #        # SHARED METHODS:
    #            classmethods:
    #                > wrap_kernel(cls, kernel, **kwargs)
    #            staticmethods:
    #                > find_arguments(kernel)
    #                > difference_mat(data1, data2)
    #            instancemethods:
    #                > kernel_setup(self)
    #        # KERNEL FUNCTIONS:
    #            classmethods:
    #                > gaussian_kernel(cls, data1, data2, length, const)
    #                > periodic_kernel(cls, data1, data2, const, period, length)
    #                > rational_quadratic(cls, data1, data2, const, length, alpha)
    # =============================================================================

    # =============================================================================
    #     # > SHARED METHODS FOR KERNEL EVALUATION
    # =============================================================================
    # > KERNEL FUNCTION WRAPPING
    @classmethod
    def wrap_kernel(cls, kernel, **kwargs):
        
        arguments = cls.find_arguments(kernel)
        argument_dict = kwargs
        arglist = tuple(argument_dict.keys())

        assert len(arguments) == len(arglist), "wrong number of arguments for kernel!"
        assert not sum(
            [not i in arguments for i in arglist]
        ), "wrong arguments have been passed to the wrapper"

        def wrapper(*args, **kwargs):
            for param, paramvalue in argument_dict.items():
                kwargs.update({param: paramvalue})

            # FUTURE: handle gradient optimization
            # when gradient optimization is implemented
            # you may want to add a differente update for gradient flag
            # kwargs.update({"wantgrad": wantgrad})

            return kernel(*args, **kwargs)

        return wrapper

    # > FIND KERNEL ARGUMENTS
    @staticmethod
    def find_arguments(kernel):
        varnames = kernel.__code__.co_varnames

        try:
            kernel_arguments = varnames[3 : varnames.index("wantgrad") + 1]
        except ValueError:
            raise Exception(f"kernel function {kernel} not valid")

        return kernel_arguments

    # > MATRIX DIFFERENCE BETWEEN VECTORS
    @staticmethod
    def difference_mat(data1, data2):

        dim1 = len(data1)
        dim2 = len(data2)

        dvec1 = np.tile(data1, (dim2, 1)).T
        dvec2 = np.tile(data2, (dim1, 1))

        diff = dvec1 - dvec2

        return diff

    # > KERNEL SETUP
    def kernel_setup(self):
        assert self.params != {}, "Kernel parameters not set!"
        
        # don't want kernels with enabled gradients to go into K, K* and K** calculations
        if self.params['wantgrad'] == True:
            self.params['wantgrad'] = False
            
        self.wrapped_kernel = self.wrap_kernel(self.kernel, **self.params)

    def update_params(self, newparams_dict):
        newparams_names = list(newparams_dict.keys())
        if not 'wantgrad' in newparams_names:
            newparams_dict['wantgrad'] = False
            newparams_names.append('wantgrad')

            
        assert len(self.find_arguments(self.kernel)) == len(
            newparams_names
        ), "wrong number of parameters for kernel!"
        assert not sum(
            [not i in newparams_names for i in self.find_arguments(self.kernel)]
        ), "you're trying to update a different list of parameters"

        self.params = newparams_dict
        self.kernel_setup()
        self.calc_kernel_matrices()

    # =============================================================================
    #     KERNEL FUNCTIONS
    #        a standard kernel function should input data1, data2 and parameters
    #        remember ALWAYS to make const the last parameter as it's position
    #        is needed when wrapping and passing arguments
    #
    #        FUTURE:
    #            when grad optimization is implemented, the grad flag will be the
    #            last argument
    # =============================================================================
    # > GAUSSIAN KERNEL
    @classmethod
    def gaussian_kernel(cls, data1, data2, length=1, const=1, wantgrad=False):
        returns = []

        square_diff = cls.difference_mat(data1, data2) ** 2

        exp = np.exp(-2 * (square_diff / np.square(length)))

        k = np.square(const) * exp
        returns.append(k)

        if wantgrad == True:
            dk_dc = 2 * const * exp
            dk_dl = (np.square(const) * square_diff * exp) / (np.power(length, 3))

            grads = {"length": dk_dl, "const": dk_dc}

            returns.append(grads)

        return np.squeeze(returns)

    # > PERIODIC KERNEL
    @classmethod
    def periodic_kernel(cls, data1, data2, period=1, length=1, const=1, wantgrad=False):
        returns = []

        dist = cls.difference_mat(data1, data2)
        sin_argument = (np.pi / period) * dist 
        squared_sin = np.square(np.sin(sin_argument)/ np.square(length))
        exponential = np.exp(-2 * squared_sin)
        k = np.square(const) * exponential
        returns.append(k)

        if wantgrad == True:
            dk_dc = 2 * const * exponential
            dk_dl = (
                4 * np.square(const) * exponential * squared_sin / np.power(length, 3)
            )
            dk_dp = (
                4
                * np.pi
                * np.square(const)
                * exponential
                * np.sin(sin_argument)
                * np.cos(sin_argument)
            ) / np.square(length * period)

            grads = {"length": dk_dl, "period": dk_dp, "const": dk_dc}
            returns.append(grads)

        return np.squeeze(returns)

    # > RATIONAL QUADRATIC KERNEL
    @classmethod
    def rational_quadratic(
        cls, data1, data2, alpha=1, length=1, const=1, wantgrad=False
    ):

        returns = []

        diff = cls.difference_mat(data1, data2)
        squared_diff = diff ** 2

        argument = 1 + squared_diff / (2 * alpha * np.square(length))
        bracket = np.power(argument, -alpha)

        k = np.square(const) * bracket
        returns.append(k)

        if wantgrad == True:
            dk_dc = 2 * const * bracket
            dk_dl = (
                np.square(const)
                * bracket
                * squared_diff
                / (np.power(length, 3) * argument)
            )
            dk_da = (
                np.square(const)
                * bracket
                * (
                    -np.log(argument)
                    + (squared_diff / (2 * np.square(length) * alpha * argument))
                )
            )

            grads = {"length": dk_dl, "const": dk_dc, "alpha": dk_da}

            returns.append(grads)

        return np.squeeze(returns)

    #    > MAUNA LOA KERNEL
    @classmethod
    def mauna_loa_example_kernel(
        cls,
        data1,
        data2,
        RBF_const=1,
        RBF_length=1,
        RBFperiodic_const=1,
        RBFperiodic_length=1,
        PERIODIC_length=1,
        RADQUAD_const=1,
        RADQUAD_length=1,
        RADQUAD_shape=1,
        RBFnoise_length=1,
        RBFnoise_const=1,
        wantgrad=False,
    ):
        returns = []

        gaussian_component, *grad_gaussiancomponent = cls.gaussian_kernel(
            data1=data1, data2=data2, length=RBF_length, const=RBF_const, wantgrad=wantgrad
        )

        periodic_component0, *grad_periodic_component0 = cls.gaussian_kernel(
            data1=data1,
            data2=data2,
            length=RBFperiodic_length,
            const=RBFperiodic_const,
            wantgrad=wantgrad,
        )

        periodic_component1, *grad_periodic_component1 = cls.periodic_kernel(
            data1=data1,
            data2=data2,
            period=1,
            length=PERIODIC_length,
            const=1,
            wantgrad=wantgrad,
        )

        rational_quadratic_component, *grad_rational_quadratic_component = cls.rational_quadratic(
            data1=data1,
            data2=data2,
            alpha=RADQUAD_shape,
            length=RADQUAD_length,
            const=RADQUAD_const,
            wantgrad=wantgrad,
        )

        noise_component, *grad_noise_component = cls.gaussian_kernel(
            data1=data1,
            data2=data2,
            length=RBFnoise_length,
            const=RBFnoise_const,
            wantgrad=wantgrad,
        )

        k = gaussian_component
#            + periodic_component0 * periodic_component1
#            + rational_quadratic_component
#            + noise_component

        returns.append(k)

        if wantgrad == True:
            grads = {'RBF_const':           grad_gaussiancomponent['const'],
                     'RBF_length':          grad_gaussiancomponent['length'],
                     'RADQUAD_const':       grad_rational_quadratic_component['const'],
                     'RADQUAD_length':      grad_rational_quadratic_component['length'],
                     'RADQUAD_shape':       grad_rational_quadratic_component['alpha'],
                     'RBFnoise_length':grad_noise_component['length'],
                     'RBFnoise_const': grad_noise_component['const'],
                     'RBFperiodic_const':   np.multiply(periodic_component1, grad_periodic_component0['const']),
                     'RBFperiodic_length':  np.multiply(periodic_component1, grad_periodic_component0['length']),
                     'PERIODIC_length':     np.multiply(periodic_component0, grad_periodic_component0['length'])}
            
            returns.append(grads)
            
        return np.squeeze(returns)
    
    @classmethod
    def mauna_loa_example_kernel2(
        cls,
        data1,
        data2,
        RBF_const=1,
        RBF_length=1,
        RBFperiodic_const=1,
        RBFperiodic_length=1,
        PERIODIC_length=1,
        RADQUAD_const=1,
        RADQUAD_length=1,
        RADQUAD_shape=1,
        RBFnoise_length=1,
        RBFnoise_const=1,
        wantgrad=False,
    ):
        
        gaussian_component = cls.gaussian_kernel(data1 = data1,
                                                 data2 = data2,
                                                 length = RBF_length,
                                                 const = RBF_const,
                                                 wantgrad = False)
        
        periodic_component1 = cls.gaussian_kernel(data1 = data1,
                                                 data2 = data2,
                                                 const = RBFperiodic_const,
                                                 length = RBFperiodic_length,
                                                 wantgrad = False)
        
        periodic_component2 = cls.periodic_kernel(data1 = data1,
                                                  data2 = data2,
                                                  const = 1,
                                                  period = 1,
                                                  length = PERIODIC_length,
                                                  wantgrad = False)
        
        rational_quadratic = cls.rational_quadratic(data1 = data1,
                                                    data2 = data2,
                                                    const = RADQUAD_const,
                                                    length = RADQUAD_length,
                                                    alpha = RADQUAD_shape,
                                                    wantgrad = False)
        
        noise_component = cls.gaussian_kernel(data1 = data1,
                                              data2 = data2,
                                              const = RBFnoise_const,
                                              length = RBFnoise_length,
                                              wantgrad = False)
        
        k = gaussian_component + np.multiply(periodic_component1, periodic_component2) + rational_quadratic + noise_component
        
        return k
    
    

    # =============================================================================
    # K, K*, K** CALCULATIONS
    #    instancemethods:
    #        > calc_K(self)
    #        > calc_Ks(self)
    #        > calc_Ks(self)
    #        > calc_kernel_matrices(self)
    # =============================================================================

    # TODO: add exceptions
    # I may need to add some exceptions for the case where no params are ready

    # NOTE: the only reason why there are three separate methods for doing
    # basically the same thing is to avoid argument confusion

    # TODO: reconsider if this kind of architecture for matrix calculation makes sense
    # to calculate K, K* and K** outside you shoud first wrap a kernel and
    # then use it with (x,x) , (x,y) and (y,y). I may change this architecture
    # in the future.

    def calc_K(self):
        return self.wrapped_kernel(self.x, self.x)

    def calc_Ks(self):
        return self.wrapped_kernel(self.x, self.x_guess)

    def calc_Kss(self):
        return self.wrapped_kernel(self.x_guess, self.x_guess)

    def calc_kernel_matrices(self):
        print(">calculating K, Ks, Kss...")
        self.kernel_setup()
        self.K = self.calc_K()
        self.Ks = self.calc_Ks()
        self.Kss = self.calc_Kss()

    # =============================================================================
    # PREDICTIONS
    #        staticmethods
    #            >calc_logp(alpha, L, y)
    #            >calc_logp_nochol(Kn, Kn_inv, y, N)
    #            >calc_gradlogp_i(Kinv, dK_di, y)
    #            > get_L_alpha(K,y)
    #                > get_Knoise(K,R,N)
    #        instancemethods
    #            >calc_logp(self)
    #            >predict(self)
    #            >predict_nochol(self)
    # =============================================================================
    @staticmethod
    def calc_logp(alpha, L, y):
        logp = (
            -0.5 * np.dot(y.T, alpha) - np.trace(L) - 0.5 * len(y) * np.log(2 * np.pi)
        )
        return logp
    
    @staticmethod
    def calc_logp_nochol(Kn,Kn_inv,y,N):
        logp = - 0.5 *  np.dot(y.T, np.dot(Kn_inv, y)) - 0.5 * np.log(np.trace(Kn)) - 0.5*N*np.log(2*np.pi)
        return logp
    
    @staticmethod
    def calc_gradlogp_i(Kinv, dK_di, y):
        gradlogp_i = 0.5*np.dot(y.T,np.dot(Kinv, np.dot(dK_di,np.dot(Kinv, y)))) - 0.5*np.trace(np.dot(Kinv, dK_di))
        
        return gradlogp_i
    
    @staticmethod
    def calc_gradlogp_i2(K, Kinv, dK, y):
        alpha = np.dot(Kinv, y)
        t = np.dot(alpha, alpha.T) - Kinv
        grad = 0.5*np.trace(np.dot(t, dK))
        
        return grad
    
    @staticmethod
    def get_L_alpha(Kn, y):
        L = np.linalg.cholesky(Kn)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return L, alpha

    @staticmethod
    def get_Knoise(K, R, N):
        return K + R * np.eye(N)
    
    
    def predict(self, alt_logp = False):
        
        
        K_noise = self.get_Knoise(self.K, self.R, self.N)
        
        L, alpha = self.get_L_alpha(K_noise, self.y)
        
        self.logp = self.calc_logp(alpha, L, self.y)

        self.pred_y = np.dot(self.Ks.T, alpha)
        

        v = np.linalg.solve(L, self.Ks)
        
        self.pred_variance = np.diag(np.diag(self.Kss) - np.dot(v.T, v))

        if self.normalize_y == True:
            self.pred_y = self.pred_y + self.y_mean

        return self.pred_y, self.pred_variance, self.logp
    
    
    def predict_nochol(self):
        K_noise = self.get_Knoise(self.K, self.R, self.N)
        
        K_noise_inv = np.linalg.inv(K_noise)
        
        self.pred_y = np.dot(self.Ks.T, np.dot(K_noise_inv, self.y))
        V = self.Kss - np.dot(self.Ks.T, np.dot(K_noise_inv, self.Ks))
        self.pred_variance = np.diag(V)
        
        self.logp = self.calc_logp_nochol(K_noise, K_noise_inv, self.y, self.N)
        
        if self.normalize_y == True:
            self.pred_y = self.pred_y + self.y_mean
            
        return self.pred_y, self.pred_variance, self.logp


#    # TO BE REFINED TO MATCH OPTIMIZER REQUESTS
#    def get_gradlogp(self, params):
#        params['wantgrad'] = True
#        
#        K, gradK = self.calc_gradK(params)
#        
#        #FOR KEY IN DICT CALCULATE STUFF
#        # but we don't really like dicts here....
#        
#        #RETURN STUFF
#    
        

    # =============================================================================
    # PLOTS
    #        staticmethods
    #            >create_figure(title, axlabelsm, axlims, figsize)
    #            >save_figure(ax,title)
    #        instancemethods
    #            >plot_process(self, mean, var, x_guess, ax)
    #            >plot_measures(self,x,y,ax)
    #            >plot(self, plot_process, plot_measures, title, save, return_ax
    #                  x,y,x_guess, pred_y, var_pred, axlims, figsize)
    # =============================================================================

    @staticmethod
    def create_figure(title, axlabels=None, axlims=None, figsize = None):
        fig, ax = plt.subplots()
        if figsize is not None:
            fig.set_figheight(figsize[1])
            fig.set_figwidth(figsize[0])
            
        ax.set_title(title)
        if axlabels is not None:
            ax.set_xlabel(axlabels[0])
            ax.set_ylabel(axlabels[1])

        if axlims is not None:
            ax.set_xlim(axlims[0])
            ax.set_ylim(axlims[1])

        return ax

    @staticmethod
    def save_figure(ax, title):
        plt.sca(ax)
        plt.savefig(title)

    def plot_process(self, mean, var, x_guess, ax):

        std_dev = np.sqrt(np.sqrt(var ** 2))
        ax.plot(x_guess, mean, label="media_processo", color="red")
        ax.fill_between(
            x_guess,
            mean - std_dev,
            mean + std_dev,
            color="lightsteelblue",
            label="std_processo",
        )

    def plot_measures(self, x, y, ax):
        ax.scatter(x, y, label="misure")

    def plot(
        self,
        plot_process=True,
        plot_measures=True,
        title="Gaussian Process Regression",
        axlabels=None,
        axlims=None,
        save=False,
        figsize = None,
        return_ax=False,
        x=None,
        y=None,
        x_guess=None,
        pred_y=None,
        var_pred=None,
        ax=None,
    ):
        """Function for plotting stuff

    Parameters
    ----------
    plot_process : bool (True)
        Enables plotting of the process after regression
    plot_measures : bool (True)
        Enables scatterplot of the measure points
    title : str ("Gaussian Process Regression")
        Sets the title of the plot
    axlabels : list of str (None)
        Labels for x and y axes
    axlims : list of tuples (None)
        Limits for x and y axes
    save : str (False)
        Filename for plot saving
    figsize : list (None)
        Sets figure size
    return_ax : bool (False)
        Returns the axes for plot
    ax: matplotlib ax (None)
        Uses an ax type for input, useful for plot integration
    x : numpy array (None)
        Explictly set measure x data
    y : numpy array (None)
        Explicitly set measure y data
    x_guess : numpy array (None)
        Explicitly set prediction points
    pred_y : numpy array (None)
        Explicitly set predictive data
    var_pred : numpy array (None)
        Explicitly set predictive variances
    y : numpy array (None)
        Explicitly set measure y data
    print_cols : bool, optional
        A flag used to print the columns to the console (default is
        False)

    Returns
    -------
    ax : matplotlib ax
        If return_ax is enabled returns ax for the plot
    """

        pred_y = pred_y if pred_y is not None else self.pred_y
        var_pred = var_pred if var_pred is not None else self.pred_variance
        x_guess = x_guess if x_guess is not None else self.x_guess
        x = x if x is not None else self.x
        y = y if y is not None else self.y_input

        ax = ax if ax is not None else self.create_figure(title, axlabels, axlims, figsize)

        if plot_process:
            self.plot_process(mean=pred_y, var=var_pred, x_guess=x_guess, ax=ax)
        if plot_measures:
            self.plot_measures(x=x, y=y, ax=ax)

        if save != False:
            self.save_figure(ax, save)

        ax.legend()
        if return_ax:
            return ax

    # =============================================================================
    #  OPTIMIZATION
    #            instancemethods
    #                > optimizer(self, ranges_dict, Ns, output_grid)
    # =============================================================================

    # > OPTIMIZER
    # at the moment is a simple brute force optimizer


    def optimizer(self, mode="brute", ranges_dict=None, param_x0 = None, Ns=100, output_grid=False):



        modes = ["brute", 'CG','Newton-CG', 'Nelder-Mead']
        assert (
            mode in modes
        ), "please select a valid mode for the optimizer, choose between: {}".format(
            *modes
        )
        
        # NOT THE MOST EFFICIENT THING EVER
        def logp(x, *args):
            params = np.exp(x)
            param_names = args

            param_dict = dict(zip(param_names, params))
            param_dict['wantgrad'] = False
            w_kernel = self.wrap_kernel(self.kernel, **param_dict)
            try:
                K = w_kernel(self.x, self.x)
                Kn = self.get_Knoise(K, self.R, self.N)
                L, alpha = self.get_L_alpha(Kn, self.y)
                logp = self.calc_logp(alpha, L, self.y)
#                logp = self.alt_calc_logp(Kn, self.y, self.N)
                
            except np.linalg.LinAlgError:
                logp = -np.inf
#                print("linalgerror")
                
#            print("logp: ", logp)
            return -logp
        
        def gradlogp(x, *args):
            params = np.exp(x)
            param_names = args
            
            grads = np.zeros(len(param_names))

            param_dict = dict(zip(param_names, params))
            
            param_dict["wantgrad"] = True
            
            w_kernel = self.wrap_kernel(self.kernel, **param_dict)
            K, gradK = w_kernel(self.x, self.x)
            Kn = self.get_Knoise(K, self.R, self.N)
            
            for i, key in enumerate(param_names):
                try:
                    dK_di = gradK[key]
                    dK_di_n = self.get_Knoise(dK_di, self.R, self.N)
                    
                    Kninv = np.linalg.inv(Kn)
                    
#                    grads[i] = -self.calc_gradlogp_i(Kninv, dK_di_n, self.y)
                    
                    grads[i] = - self.calc_gradlogp_i2(K, Kninv, dK_di_n, self.y)

                except np.linalg.LinAlgError:
#                    print("hey errore")
                    grads[i] = np.inf

#            print("grads: ", grads)
            return grads
            
        def extract_results(minres, parm_names):
            res = np.exp(minres.x)
            logp = -minres.fun
            return dict(zip(param_names, res)), logp
        
        if mode == "brute":
            returns  = []
            param_names = tuple(ranges_dict.keys())
            param_ranges = tuple(ranges_dict.values())
            
            x0, fval, grid, Jout = brute_optim(
                func=logp,
                ranges=param_ranges,
                args=(param_names),
                Ns=Ns,
                full_output=True,
                disp=True,
            )

            optim_params = dict(zip(param_names, x0))
            returns.append(optim_params)
            returns.append(-fval)

            if output_grid:
                returns.append(grid)
                returns.append(Jout)
                
            return returns
        
        
        elif mode in modes[1:]:
            param_names = tuple(param_x0.keys())
            param_x0 = tuple(param_x0.values())
            
            if mode == "CG" or mode == "Newton-CG":
                jac = gradlogp
            else:
                jac = None
            
            
            min_results = fmin(fun = logp,
                           x0 = param_x0,
                           args = (param_names),
                           method = mode,
                           jac = jac,
                           options = {'disp': True},
                           tol = 1e-3)
            
            return extract_results(min_results, param_names)
            
