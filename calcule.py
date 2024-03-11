import datetime
import math
import time
import warnings
import numpy as np
import pyqtgraph as pg
from numba import njit
from numba.core import config
from scipy.special import erfinv

import gg02
import utilities
from utilities import *
import coding
import galois
import  ldpc
from timeit import *
from ldpc import *
import preprocessing
from gg02 import *
from preprocessing import normalization
is_mu_optimal = False  # Determines if the modulation variance will be optimised automatically or set manually
is_error_corrected = False  # Determines if error correction takes place or only the composable key rate is computed
is_code_loaded = False
# Input values definition

L = 5  # Channel length (km)
A = 0.2  # Attenuation rate (dB/km)
xi = 0.01 #Excess noise
eta = 0.8  # Detector/Setup efficiency
v_el = 0.1  # Electronic noise
beta = 0.97  # Reconciliation parameter
n_bks =5 # Number of blocks
N = 500000# Block size
M = (n_bks * N) * 0.1  # Number of PE runs
p = 6  # Discretization bits
q = 4  # Most significant (top) bits
alpha = 7  # Phase-space cut-off
iter_max = 40  # Max number of EC iterations
e_PE = 2 ** -32  # Probability that the estimated parameters do not belong in the confidence region
e_s = 2 ** -32  # Smoothing parameter
e_h = 2 ** -32  # Hashing parameter
e_cor = 2 ** -32  # Correctness error (universal hash function collision probability)
p_EC_tilde = 0.99
T = 10 ** (-A * L / 10)  # Channel Losses (dB)
s_z = 1 + v_el + eta * T * xi  # Noise variance
Xi = eta * T * xi  # Excess noise variance
Chi = xi + (1 + v_el) / (T * eta)  # Equivalent noise
m = int(M / n_bks)  # PE instances per block
n = N - m  # Key generation points per block
t = int(np.ceil(-np.log2(e_cor)))  #Verification hash output length
GF = 2 ** q  # Number of the Galois Field elements
delta = alpha / (2 ** (p - 1))  # Lattice step in phase space
d = p - q  # Least significant (bottom) bits
if is_mu_optimal:  # For a fixed reconciliation parameter β, find the optimal modulation variance μ >= 1
    mu = optimal_modulation_variance(T, eta, xi, v_el, beta)
else:
    mu = 21.89226929460376
def depedent_values(mu,L,A,xi,eta,v_el,n_bks,N,p,q):
   T = 10 ** (-A * L / 10)  # Channel Losses (dB)
   s_z = 1 + v_el + eta * T * xi  # Noise variance
   Xi = eta * T * xi  # Excess noise variance
   Chi = xi + (1 + v_el) / (T * eta)  # Equivalent noise
   M = (n_bks * N) * 0.1  # Number of PE runs
   m = int(M / n_bks)  # PE instances per block
   n = N - m  # Key generation points per block
   t = int(np.ceil(-np.log2(e_cor)))  # Verification hash output length
   GF = 2 ** q  # Number of the Galois Field elements
   delta = alpha / (2 ** (p - 1))  # Lattice step in phase space
   d = p - q  # Least significant (bottom) bits
   SNR = (mu - 1) / Chi  # Signal-to-noise ratio
   return T,s_z,Xi,Chi,SNR
#T, s_z, Xi, Chi, SNR = depedent_values(mu, L, A, xi, eta, v_el, n_bks, N, p, q)
##########################################################################################

def secure_key_rate_graphe(L,xi,eta,v_el,beta,n_bks,N,q,e_PE):
   # Determines if the modulation variance will be optimised automatically or set manually
   T,s_z,Xi,Chi,SNR = depedent_values(mu, L, A, xi, eta, v_el, n_bks, N, p, q)

   # Alice prepares and transmits the coherent states. Bob receives the noisy states and measures them. After measuring,
   # they perform key sifting.
   X = np.empty(shape=[n_bks, N], dtype=np.float64)  # Alice's variable
   Y = np.empty(shape=[n_bks, N], dtype=np.float64)  # Bob's variable

   for blk in range(n_bks):
      Q_X, P_X = prepare_states(N, mu)
      Q_Y, P_Y = transmit_states(N, Q_X, P_X, T, eta, s_z)
      qu, Y[blk] = measure_states(N, Q_Y, P_Y)
      X[blk] = key_sifting(N, Q_X, P_X, qu)

   # Calculate the asymptotic key rate
   I_AB, x_Ey, R_asy = key_rate_calculation(mu, T, eta, xi, v_el, beta)

   # Determine the states for key generation and parameter estimation and perform parameter estimation
   X_key, Y_key, X_PE, Y_PE = sacrificed_states_selection(n_bks, n, m, M, X, Y)
   T_hat, xi_hat, T_m, xi_m, T_star_m, xi_star_m = parameter_estimation(mu, X_PE.ravel(), Y_PE.ravel(), T, eta,
                                                                        Xi,
                                                                        v_el, M, s_z, e_PE)

   # In the next step, they compute an overestimation of the Holevo bound in terms of T_m and ξ_m, so that they may write
   # the modified rate
   I_AB_hat, _, _ = key_rate_calculation(mu, T_hat, eta, xi_hat, v_el, beta)
   _, x_M, _ = key_rate_calculation(mu, T_m, eta, xi_m, v_el, beta)
   R_M = beta * I_AB_hat - x_M

   # The theoretical worst-case Holevo bound is calculated using the theoretical estimators
   _, x_M_star, _ = key_rate_calculation(mu, T_star_m, eta, xi_star_m, v_el, beta)
   R_M_star = beta * I_AB - x_M_star

   # Bob checks the threshold condition I(x : y|T^,ξ^ > χ(E : y)|TM,ξM. If it is not satisfied, the protocol is aborted.
   if I_AB_hat <= x_M:
      raise RuntimeWarning(
         "Estimated mutual information is smaller than worst-case Holevo bound. Protocol is aborted.")
   # Accounting for the number of signals sacrificed for parameter estimation, the actual rate in terms of bits per
   # channel use is given by the rescaling
   R_m = (n / N) * R_M

   # Perform EC preprocessing, i.e., normalization, discretization and splitting of the key generation sequences
   K = np.empty(shape=[n_bks, n], dtype=np.int16)  # Bob's quantized sequence
   K_bar = np.empty(shape=[n_bks, n], dtype=np.int16)  # Bob's most significant bits to be used in encoding
   K_ubar = np.empty(shape=[n_bks, n], dtype=np.int16)  # Bob's least significant bits to be sent in the clear
   P = np.empty(shape=[n_bks, n, 2 ** q], dtype=np.float64)  # The a-priori probabilities for error correction
   field_values = np.arange(2 ** q)  # All possible values that belong in the specified Galois field

   X_key, Y_key = preprocessing.normalization(X_key, Y_key)
   SNR_hat, rho, rho_th = code_estimations(mu, X_key, Y_key, T_hat, eta, v_el, xi_hat)

   for blk in range(n_bks):
      K[blk] = preprocessing.discretization(Y_key[blk], alpha, p, delta)
      K_bar[blk], K_ubar[blk] = preprocessing.splitting(K[blk], d)
      P[blk] = coding.a_priori_probabilities(X_key[blk], K_ubar[blk], field_values, rho, alpha, p, delta, d)

   # Identify the rate of the error-correcting code
   R_code, R_code_approx, H_K = code_rate_calculation(K, n_bks, n, beta, I_AB_hat, p, q, alpha, SNR_hat)
   p_EC, FER = frame_error_rate_calculation(n_bks * p_EC_tilde, n_bks)
   R, R_theo, n_tilde, r, epsilon = composable_key_rate(n_bks, N, n, p, q, R_code,
                                                        R_M_star, x_M, p_EC, e_s, e_h,
                                                        e_cor, e_PE, H_K)
   return SNR_hat,p_EC,FER,H_K,x_M,x_Ey,R_asy, R_M, R_m, I_AB,\
          I_AB_hat,R_code, epsilon,R

######################################################################################################


SNR_hat,p_EC,FER,H_K,x_M,x_Ey,R_asy, R_M, R_m, I_AB,\
          I_AB_hat,R_code, epsilon,R = secure_key_rate_graphe(L,xi, eta, v_el, beta, 5, 500000, q, e_PE)
###############################################################################################

#print(X_PE)