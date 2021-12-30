# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import datasets
from scipy.optimize import minimize



                                        
class DReaM:
    def __init__(self, X, Y, K = 2, 
            alpha_t = .1, beta_t = 1., a = 10., cov = "full", 
            mu_t_plus0 = None, mu_t_minus0 = None, 
            ):
        """
        Constructor for DReaM.
        X: Rule-generating features.
        Y: Cluster-preserving features.
        alpha_t, beta_t, a: Model parameters. Read the paper for details.
        mu_t_plut0, mu_t_minus0 defines the prior rules.
        """
            
        [self.N, self.D] = X.shape

        self.X_shift = X.mean(0) 
        self.X_scale = X.std(0) + 1e-20
        
        #Normalize the data
        self.X = (X - self.X_shift) / self.X_scale


        [self.N, self.D_Y] = Y.shape

        
        self.Y_shift = Y.mean(0) 
        self.Y_scale = Y.std(0) + 1e-20
        self.Y = (Y - self.Y_shift) / self.Y_scale
        
            
                        
        self.K = K
        
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        
        initialization = "GMM"
        
        #If prior rule is provided, initialize with the prior rules.                
        if mu_t_plus0 is None or (mu_t_plus0 == 0).all():
            self.mu_t_plus0 = np.zeros([self.K, self.D])
        else:
            initialization = "boundary"
            self.mu_t_plus0 = ( mu_t_plus0 - self.X_shift) / self.X_scale
            
        if mu_t_minus0 is None or (mu_t_minus0 == 0).all():
            self.mu_t_minus0 = np.zeros([self.K, self.D])
        else:
            initialization = "boundary"
            self.mu_t_minus0 = ( mu_t_minus0 - self.X_shift) / self.X_scale
            
                
        self.a = a
        self.cov = cov

        self.initialize(initialization)
        
    def log_likelihood(self):
        """
        Computing the log_likelihood.
        """
        res = 0
        for k in range(self.K):
            T_k = np.concatenate([self.t_plus_kd[k], self.t_minus_kd[k]])
            res += - self.neg_Q(T_k, k)

            if self.cov == "full":
                res += np.dot(self.pi_nk[:, k], 
                    - .5 * np.linalg.slogdet(self.Sigma_k[k])[1]
                    - .5 * np.einsum("ni, ij, nj -> n", self.Y - self.mu_k[k], self.Sigma_k_I[k], self.Y - self.mu_k[k])
                )
            else:
                res += np.dot(self.pi_nk[:, k], 
                    - .5 * np.log(self.sigma2_kd[k] + 1e-300)
                    - .5 * ( self.Y - self.mu_k[k] ) ** 2 / (self.sigma2_kd[k] + 1e-300)
                ).sum()
            
        res += - (self.pi_nk * np.log(self.pi_nk + 1e-300)).sum()
        
        return res
    
    def initialize(self, initialization = "GMM"):
        """
        Initialize the model with either GMM or the prior rules.
        """
        self.mu_k = np.zeros([self.K, self.D_Y])
        
        if self.cov == "full":
            self.Sigma_k = np.zeros([self.K, self.D_Y, self.D_Y])
            self.Sigma_k_I = np.zeros([self.K, self.D_Y, self.D_Y])
        else:
            self.sigma2_kd = np.zeros([self.K, self.D_Y])
            
            
        if initialization == "boundary":
            gamma_nk = np.zeros([self.N, self.K])
            self.t_plus_kd = self.mu_t_plus0.copy()
            self.t_minus_kd = self.mu_t_minus0.copy()
            
            for k in range(self.K):
                sigmoid_plus = 1. / (1 + np.exp(- self.a * (self.t_plus_kd[k,:] - self.X) ))
                sigmoid_minus = 1. / (1 + np.exp(- self.a * (self.X - self.t_minus_kd[k,:]) ))
        
                gamma_nk[:,k] = sigmoid_plus.prod(1) * sigmoid_minus.prod(1)

            self.pi_nk = np.log(gamma_nk + 1e-300) - np.log(1 - gamma_nk + 1e-300)
            self.pi_nk += np.random.normal(0, self.pi_nk.std() / 10, [self.N, self.D])

            self.pi_nk = (self.pi_nk.T - self.pi_nk.max(1)).T
            
            self.pi_nk = np.exp(self.pi_nk)
            self.pi_nk = (self.pi_nk.T / self.pi_nk.sum(1)).T
        
        elif initialization == "GMM":
            M = GaussianMixture(n_components = self.K, covariance_type = self.cov, n_init = 1)
            M.fit(self.Y)
            
            self.pi_nk = np.zeros([self.N, self.K])
            self.pi_nk[range(self.N), M.predict(self.Y)] = 1

            mean_ = np.zeros([self.K, self.D])
            std_ = np.zeros([self.K, self.D])
        
            for k in range(self.K):
                N_k = self.pi_nk[:,k].sum()
                mean_[k,:]= np.dot(self.pi_nk[:,k].T, self.X) / N_k
                std_[k,:] = np.sqrt(np.dot(self.pi_nk[:,k], ((self.X[:] - mean_[k])**2)) / N_k )
            
            self.t_plus_kd = mean_ + std_
            self.t_minus_kd = mean_ - std_
                        
                                                                        
                                                
        
        self.mu_k = np.einsum("nk, nd -> kd", self.pi_nk, self.Y)
        self.mu_k = (self.mu_k.T / self.pi_nk.sum(0)).T
        
        if self.cov == "full":
            for k in range(self.K):
                self.Sigma_k[k] = np.einsum("n, ni, nj -> ij", 
                    self.pi_nk[:, k], self.Y - self.mu_k[k], self.Y - self.mu_k[k])
                self.Sigma_k[k] = self.Sigma_k[k] / self.pi_nk[:, k].sum()  + 1e-5 * np.eye(self.D_Y)
                self.Sigma_k_I[k] = np.linalg.inv(self.Sigma_k[k])
        else:
            for k in range(self.K):
                self.sigma2_kd[k] = np.einsum("n, nd -> d", 
                    self.pi_nk[:, k], (self.Y - self.mu_k[k]) ** 2)
                self.sigma2_kd[k] = self.sigma2_kd[k] / ( self.pi_nk[:, k].sum()  + 1e-5 )
            
            
            
        
    def repeat(self, n_init = 10, n_iter = 100):
        """
        Repeat the algorithm for several times.  
        We keep the results with the maximum likelihood.
        n_init: Number of initializations.
        n_iter: Number of iteration for the EM algorithm in each run.
        """
        results = None
        max_lowerbound = -1e300
        for ii in range(n_init):
            print("repeat {} / {}".format(ii, n_init))
            
            if (self.mu_t_plus0 == 0).all() and (self.mu_t_minus0 == 0).all():
                self.initialize("GMM")
            else:
                self.initialize("boundary")
                
            self.EM(n_iter)
            if self.log_likelihood()>max_lowerbound:
                max_lowerbound = self.log_likelihood()
                if self.cov == "full":
                    results = [self.t_plus_kd, self.t_minus_kd, self.pi_nk, self.mu_k, self.Sigma_k, self.Sigma_k_I]
                else:
                    results = [self.t_plus_kd, self.t_minus_kd, self.pi_nk, self.mu_k, self.sigma2_kd]
        
        if self.cov == "full":
            [self.t_plus_kd, self.t_minus_kd, self.pi_nk, self.mu_k, self.Sigma_k, self.Sigma_k_I] = results
        else:
            [self.t_plus_kd, self.t_minus_kd, self.pi_nk, self.mu_k, self.sigma2_kd] = results

    def neg_diff_Q(self, T_k, k):
        """
        Return the gradient of the expected value of the log likelihood,
        i.e., the gradient of the objective function for the maximization step.
        """
        res = np.zeros(self.D * 2)
        
        t_plus_d = T_k[:self.D]
        t_minus_d = T_k[self.D:]
        
        
        sigmoid_plus = 1. / (1 + np.exp(- self.a * ( t_plus_d - self.X ) ))
        sigmoid_minus = 1. / (1 + np.exp(- self.a * ( self.X - t_minus_d ) ))

        gamma_n = sigmoid_plus.prod(1) * sigmoid_minus.prod(1)
        
        res[:self.D] = - self.alpha_t * t_plus_d + self.alpha_t * self.mu_t_plus0[k, :] \
            - self.beta_t * t_plus_d + self.beta_t * t_minus_d \
            + self.a * np.einsum("n, nd -> d", 
                self.pi_nk[:, k], 1. / (1 + np.exp( self.a * ( t_plus_d - self.X ) ))) \
            - self.a * np.einsum("n, nd, nd, n -> d", 
                1 - self.pi_nk[:, k], np.exp(- self.a * (t_plus_d - self.X) ), sigmoid_plus, gamma_n / (1 - gamma_n) )

        res[self.D:] = - self.alpha_t * t_minus_d + self.alpha_t * self.mu_t_minus0[k, :] \
            - self.beta_t * t_minus_d + self.beta_t * t_plus_d \
            - self.a * np.einsum("n, nd -> d", 
                self.pi_nk[:, k], 1. / (1 + np.exp( self.a * ( self.X - t_minus_d ) ))) \
            + self.a * np.einsum("n, nd, nd, n -> d", 
                1 - self.pi_nk[:, k], np.exp(- self.a * (self.X - t_minus_d) ), sigmoid_minus, gamma_n / (1 - gamma_n) )

                        
        return -res

    def neg_Q(self, T_k, k):
        """
        Return the expected value of the log likelihood, i.e., the objective 
        function for the maximization step.
        """
        res = 0
        
        t_plus_d = T_k[:self.D]
        t_minus_d = T_k[self.D:]
        
        
        res += - .5 * self.alpha_t * ( (t_minus_d - self.mu_t_minus0) ** 2 ).sum()
        res += - .5 * self.alpha_t * ( (t_plus_d - self.mu_t_plus0) ** 2 ).sum()
        res += - .5 * self.beta_t * ( (t_plus_d - t_minus_d) ** 2 ).sum()

        
        sigmoid_plus = 1. / (1 + np.exp(- self.a * ( t_plus_d - self.X ) ))
        sigmoid_minus = 1. / (1 + np.exp(- self.a * ( self.X - t_minus_d ) ))

        gamma_n = sigmoid_plus.prod(1) * sigmoid_minus.prod(1)
        
        res += np.dot(self.pi_nk[:, k], np.log(sigmoid_plus).sum(1) + np.log(sigmoid_minus).sum(1))
        res += np.dot(1 - self.pi_nk[:,k], np.log(1 - gamma_n))

        
        return -res

    
    def EM(self, n_iter = 100, tol = 1e-5):
        """
        Expectation Maximization (EM) algorithm.
        n_iter: maximum number of iterations.
        """
        L_0 = self.log_likelihood()
        
        for ii in range(n_iter):
            # M step
            for k in range(self.K):
                T_k = np.concatenate([self.t_plus_kd[k], self.t_minus_kd[k]])
                res = minimize(self.neg_Q, T_k, args = k, jac = self.neg_diff_Q, method = "BFGS")
                self.t_plus_kd[k] = res.x[:self.D]
                self.t_minus_kd[k] = res.x[self.D:]

            self.mu_k = np.einsum("nk, nd -> kd", self.pi_nk, self.Y)
            self.mu_k = (self.mu_k.T / self.pi_nk.sum(0)).T
            
            if self.cov == "full":
                for k in range(self.K):
                    self.Sigma_k[k] = np.einsum("n, ni, nj -> ij", 
                        self.pi_nk[:, k], self.Y - self.mu_k[k], self.Y - self.mu_k[k])
                    self.Sigma_k[k] = self.Sigma_k[k] / self.pi_nk[:, k].sum()  + 1e-5 * np.eye(self.D_Y)
                    self.Sigma_k_I[k] = np.linalg.inv(self.Sigma_k[k])
            else:
                for k in range(self.K):
                    self.sigma2_kd[k] = np.einsum("n, nd -> d", 
                        self.pi_nk[:, k], (self.Y - self.mu_k[k]) ** 2)
                    self.sigma2_kd[k] = self.sigma2_kd[k] / ( self.pi_nk[:, k].sum()  + 1e-5 )
                
            
            # E step                                    
            gamma_nk = np.zeros([self.N, self.K])
            
            for k in range(self.K):
                sigmoid_plus = 1. / (1 + np.exp(- self.a * ( self.t_plus_kd[k,:] - self.X ) ))
                sigmoid_minus = 1. / (1 + np.exp(- self.a * ( self.X - self.t_minus_kd[k,:] ) ))
        
                gamma_nk[:,k] = sigmoid_plus.prod(1) * sigmoid_minus.prod(1)

            log_pi_nk = np.log(gamma_nk + 1e-300) - np.log(1 - gamma_nk + 1e-300)
            
            if self.cov == "full":
                for k in range(self.K):
                    log_pi_nk[:, k] += - .5 * np.linalg.slogdet(self.Sigma_k[k])[1] \
                        - .5 * np.einsum("ni, ij, nj -> n", self.Y - self.mu_k[k], self.Sigma_k_I[k], self.Y - self.mu_k[k])
            else:
                for k in range(self.K):
                    log_pi_nk[:, k] += ( - .5 * np.log( self.sigma2_kd[k, :] + 1e-300 )
                        - .5 * ( ( self.Y - self.mu_k[k]) ** 2 ) / (self.sigma2_kd[k, :] + 1e-5)  ).sum(1)
                
            
            log_pi_nk = (log_pi_nk.T - log_pi_nk.max(1)).T
            
            self.pi_nk = np.exp(log_pi_nk)
            self.pi_nk = (self.pi_nk.T / self.pi_nk.sum(1)).T


            L = self.log_likelihood()    
            print ("iter {} / {}, log-likelihood:{}".format(ii, n_iter, L))
            
            if (L - L_0) < tol * np.abs(L_0):
                print ("Converged!")
                break
            else:
                L_0 = L


    def plot_rules(self, D0 = 0, D1 = 1, description = None):
        """
        Plot the rules.
        D0, D1 are the indices for the features being ploted.
        """
        markers = iter(['bs', 'gv', 'r^', 'yo', 'k.', 'c<', 'm>', 'bs', 'gv'])
        
        plt.figure(figsize = [12 , 8])
        
        font = {'weight' : 'light',
        'size'   : 20}
    
        plt.rc('font', **font)                                 
                        
        z = self.rectangle_results()
    
        for k in range(self.K):
            marker = next(markers)
            
            plt.plot(self.X[z == k, D0] * self.X_scale[D0] + self.X_shift[D0], 
                    self.X[z == k, D1] * self.X_scale[D1] + self.X_shift[D1], 
                    marker, markersize = 12, label = "Cluster{}".format(k+1))
    
            plt.plot([
                    self.t_plus_kd[k, D0]  * self.X_scale[D0] + self.X_shift[D0], 
                    self.t_minus_kd[k, D0] * self.X_scale[D0] + self.X_shift[D0]
                    ],  
                    [
                    self.t_minus_kd[k, D1]  * self.X_scale[D1] + self.X_shift[D1], 
                    self.t_minus_kd[k, D1]  * self.X_scale[D1] + self.X_shift[D1]
                    ], marker[0], linewidth=2.0)

            plt.plot([
                    self.t_plus_kd[k, D0]  * self.X_scale[D0] + self.X_shift[D0], 
                    self.t_minus_kd[k, D0] * self.X_scale[D0] + self.X_shift[D0]
                    ],
                    [
                    self.t_plus_kd[k, D1] * self.X_scale[D1] + self.X_shift[D1], 
                    self.t_plus_kd[k, D1] * self.X_scale[D1] + self.X_shift[D1]
                    ], marker[0], linewidth=2.0)
                        
            plt.plot([  
                    self.t_plus_kd[k, D0] * self.X_scale[D0] + self.X_shift[D0], 
                    self.t_plus_kd[k, D0] * self.X_scale[D0] + self.X_shift[D0]
                    ], 
                    [
                    self.t_plus_kd[k, D1] * self.X_scale[D1] + self.X_shift[D1],
                    self.t_minus_kd[k, D1]* self.X_scale[D1] + self.X_shift[D1]
                    ], marker[0], linewidth=2.0)
                    
            plt.plot(   [self.t_minus_kd[k, D0] * self.X_scale[D0] + self.X_shift[D0], 
                    self.t_minus_kd[k, D0] * self.X_scale[D0] + self.X_shift[D0]], 
                    [
                    self.t_plus_kd[k, D1] * self.X_scale[D1] + self.X_shift[D1], 
                    self.t_minus_kd[k, D1] * self.X_scale[D1] + self.X_shift[D1]
                    ], marker[0], linewidth=2.0)
                    
        
        plt.xlabel("$X_{}$".format(D0 + 1))
        plt.ylabel("$X_{}$".format(D1 + 1))
        
        plt.legend()
                    
        

                        
    def plot_Y(self, D0 = 0, D1 = 1):
        """
        plot the cluster-preserving features Y.
        D0, D1 are the indices for the features being ploted.
        """
        markers = iter(['bs', 'gv', 'r^', 'yo', 'k.', 'c<', 'm>', 'bs', 'gv'])
        
        plt.figure(figsize = [12 , 8])
        
        font = {'weight' : 'light',
        'size'   : 20}
    
        plt.rc('font', **font)                                 
                                
        z = self.rectangle_results()
        
    
        for k in range(self.K):
            plt.plot(self.Y[z == k, D0] * self.Y_scale[D0] + self.Y_shift[D0], 
                    self.Y[z == k, D1] * self.Y_scale[D1] + self.Y_shift[D1], 
                    next(markers), markersize = 12, label = "Cluster{}".format(k + 1))
    

        plt.xlabel("$Y_{}$".format(D0 + 1))
        plt.ylabel("$Y_{}$".format(D1 + 1))
        
        plt.legend(loc = 0)
        
        
    def rectangle_results(self, X = None):
        """
        Return the clustering indicators based on the rules.
        """
        if X is None:
            X = self.X

        N = X.shape[0]
        
        gamma_nk = np.zeros([N, self.K])
        
        for k in range(self.K):
            sigmoid_plus = 1. / (1 + np.exp(- self.a * ( self.t_plus_kd[k,:] - X ) ))
            sigmoid_minus = 1. / (1 + np.exp(- self.a * ( X - self.t_minus_kd[k,:] ) ))
    
            gamma_nk[:,k] = sigmoid_plus.prod(1) * sigmoid_minus.prod(1)

        log_pi_nk = np.log(gamma_nk + 1e-300) - np.log(1 - gamma_nk + 1e-300)

        z = np.argmax(log_pi_nk, 1)
        
        return z          
        
    def get_rules(self):
        """
        Return the rules as text.
        """
        text = ""
        
        for k in range(self.K):
            text += "Rules for Cluster{}:\n".format(k + 1)
            for d in range(self.D):
                text += "\t {:0.2f} < x{} < {:0.2f}\n".format(
                    self.t_minus_kd[k,d] * self.X_scale[d] + self.X_shift[d], d + 1,
                    self.t_plus_kd[k,d] * self.X_scale[d] + self.X_shift[d]
                    )
            text += "\n"
        
        return text