
import continuous
import hashes
import numpy as np
import warnings
from scipy.special import erfinv
from numba import njit,jit

def optimal_modulation_variance(t, h, x, v, b):
    mu_range = np.arange(2, 500, 0.001)  # Identify the optimal mu
    R_opt = np.zeros_like(mu_range, dtype=np.float64)  # Optimized key rate for a fixed reconciliation efficiency β
    for j in range(len(mu_range)):
        mut, hol, R_opt[j] = key_rate_calculation(mu_range[j], t, h, x, v, b)
    mu = mu_range[np.argmax(R_opt)]  # Identify the index where the key rate is maximum and use it to get the optimal μ
    return mu


@njit(fastmath=True, parallel=True, cache=True)
def prepare_states(n, m):

    q = np.random.normal(0, np.sqrt(m - 1), n)  # Q-quadrature Gaussian distribution with μ = 0 and σ^2 = μ - 1
    p = np.random.normal(0, np.sqrt(m - 1), n)  # P-quadrature Gaussian distribution with μ = 0 and σ_^2 = μ - 1

    return q, p



def transmit_states(n, q, p, t, h, s):
    qB = np.empty(n, dtype=np.float64)
    pB = np.empty(n, dtype=np.float64)
    z = np.random.normal(0, np.sqrt(s), n)  # Total noise term following a centered normal distribution

    # Le résultat du détecteur homodyne satisfait la formule d'entrée-sortie
    for i in range(n):
        qB[i] = np.sqrt(t * h) * q[i] + z[i]
        pB[i] = np.sqrt(t * h) * p[i] + z[i]

    return qB, pB


@njit(fastmath=True, parallel=True, cache=True)
def measure_states(n, q, p):
    meas_q = np.empty(n, dtype=np.uint8)  # Measured quadratures
    vals_q = np.empty(n, dtype=np.float32)  # Saved values from the measured quadratures
    b = np.random.randint(low=0, high=2, size=n)
    for i in range(n):
        if b[i] == 0:
            meas_q[i] = 0
            vals_q[i] = q[i]
        else:
            meas_q[i] = 1
            vals_q[i] = p[i]
    return meas_q, vals_q


@njit(fastmath=True, cache=True)
def key_sifting(n, q, p, y):
    x = np.empty(n, dtype=np.float32)  # Alice's measured quadratures
    # Alice keeps only the value of the quadrature that Bob measured
    for i in range(n):
        if y[i] == 0:
            x[i] = q[i]
        else:
            x[i] = p[i]
    return x


def sacrificed_states_selection(n_bks, n, m, M, x, y):
    # Alice and Bob declare m random instances {xi} and {yi} of their local variables x and y.
    x_PE = np.empty(shape=[n_bks, m], dtype=np.float64)
    y_PE = np.empty(shape=[n_bks, m], dtype=np.float64)
    x_key = np.empty(shape=[n_bks, n], dtype=np.float64)
    y_key = np.empty(shape=[n_bks, n], dtype=np.float64)
    tempX = np.copy(x)
    tempY = np.copy(y)

    # The chosen instances for parameter estimation are sacrificed and will not be used for key generation
    # On average m = M / n_bks positions are picked in each block, where n = N − m points remain for key generation
    for blk in range(n_bks):
        # Alice randomly picks m positions from each block
        # Using a classical channel, Alice communicates the M pairs to Bob
        seed = np.random.randint(1000)  # Shuffle both sequences in the same way
        np.random.seed(seed)
        np.random.shuffle(tempX[blk])
        np.random.seed(seed)
        np.random.shuffle(tempY[blk])
        x_PE[blk] = tempX[blk][:m]
        y_PE[blk] = tempY[blk][:m]
        x_key[blk] = tempX[blk][m:]
        y_key[blk] = tempY[blk][m:]

    # Ensure the number of states selected is the same as the chosen M
    assert int(x_PE.size) == M
    assert int(y_PE.size) == M
    assert int(x_key.size) == n * n_bks
    assert int(y_key.size) == n * n_bks

    return x_key, y_key, x_PE, y_PE


def parameter_estimation(vA, x, y, t, h, v_x, v, m, s, e_pe):
    t_hat = np.sum(x * y) / np.sum(x ** 2)  # Maximum-likelihood estimator of the square root transmissivity
    s_hat = (1 / m) * np.sum((y - (t_hat * x)) ** 2)  # Maximum-likelihood estimator of the noise variance

    # For m sufficiently large we have that, up to an error probability, the channel parameters falls in the intervals
    w = np.sqrt(2) * erfinv(1 - e_pe)
    D_t = w * np.sqrt(s_hat / (m * (vA - 1)))
    D_s = w * s_hat * np.sqrt(2) / np.sqrt(m)

    if np.sqrt(t * h) < t_hat - D_t or np.sqrt(t * h) > t_hat + D_t:
        warnings.warn("Root of τη does not fall under the confidence interval.")
    if s < s_hat - D_s or s > s_hat + D_s:
        warnings.warn("Noise variance σ_z does not fall under the confidence interval.")

    # Because the parties perfectly know the values of the detector/setup efficiency and the electronic noise, they may
    # derive the estimators for the transmissivity and excess noise
    T_hat = (t_hat ** 2) / h
    X_hat = s_hat - 1 - v

    if T_hat <= 0 or T_hat > 1:
        raise RuntimeWarning("The estimated transmittance must be a number between 0 and 1.")
    if X_hat < 0:
        raise RuntimeWarning("The estimated excess noise variance must be positive or zero (current:", X_hat, ")")

    # Then they may assume the worst case estimators
    T_m = ((t_hat - D_t) ** 2) / h
    X_m = X_hat + D_s

    if t < T_m:  # Up to an error ε
        warnings.warn("The actual transmittance is smaller than the worst-case estimator.")
    elif v_x > X_m:
        warnings.warn("The excess noise variance is larger than the worst-case estimator.")

    # Calculate the theoretical worst-case estimators, which are derived from the actual values
    T_star_m = (np.sqrt(t) - w * np.sqrt(s / (m * h * (vA - 1)))) ** 2
    X_star_m = v_x + w * s * np.sqrt(2 / m)

    # Obtain the excess noises from the respective variances of excess noise
    x_hat = X_hat / (h * T_hat)
    x_m = X_m / (h * T_m)
    x_star_m = X_star_m / (h * t)

    return T_hat, x_hat, T_m, x_m, T_star_m, x_star_m

def key_rate_calculation(vA, t, h, e, v, bt):
    # Calculate the thermal noise
    omega = (t * e - t + 1) / (1 - t)

    # Calculate the mutual information and the reconciliation efficiency
    s_y = t * h * (vA - 1 + e) + 1 + v  # Variance of y
    s_xy = h * t * e + 1 + v  # Conditional variance between x and y
    i = 0.5 * np.log2(s_y / s_xy)

    # Define the global output state parameters
    # c = np.sqrt(t * hd * (vA ** 2 - 1))
    b = t * h * (vA + e) + 1 - (t * h) + v
    gamma = np.sqrt(h * (1 - t) * (omega ** 2 - 1))
    # delta = -np.sqrt((1 - t) * (vA ** 2 - 1))
    theta = np.sqrt(h * t * (1 - t)) * (omega - vA)
    psi = np.sqrt(t * (omega ** 2 - 1))
    phi = t * omega + (1 - t) * vA

    v_eE = np.array([[omega, 0, psi, 0], [0, omega, 0, -psi], [psi, 0, phi, 0], [0, -psi, 0, phi]])
    v_eEy = v_eE - (b ** -1) * np.array([[gamma ** 2, 0, gamma * theta, 0], [0, 0, 0, 0],
                                         [gamma * theta, 0, theta ** 2, 0], [0, 0, 0, 0]])
    v_1, v_2 = continuous.symplectic_eigenvalue_calculation(v_eE)
    v_3, v_4 = continuous.symplectic_eigenvalue_calculation(v_eEy)

    x = continuous.h_f(v_1) + continuous.h_f(v_2) - continuous.h_f(v_3) - continuous.h_f(v_4)  # Holevo bound
    r = bt * i - x  # Secret key rate
    return i, x, r

def code_rate_calculation(k, n_bks, n, b, i, p, q, a, snr):

    unique_y, counts_y = np.unique(k.ravel(), return_counts=True)
    p_k = counts_y / (n_bks * n)  # Appearance probabilities of the discretized values
    h_k = -np.sum(p_k * np.log2(p_k))  # Shannon entropy of the discretized values of the key generation states
    r = (b * i + p - h_k) / q  # Practical code rate
    r_th = np.log2(a * np.sqrt((2 * (1 + snr) ** b) / (np.pi * np.e))) / q  # Theoretical code rate

    # The theoretical and practical rates must be somewhat close
    np.testing.assert_almost_equal(r, r_th, decimal=1)

    return r, r_th, h_k


def code_estimations(m, x, y, t, h, v, e):

    # Calculate the signal-to-noise ratio using the estimators
    snr = ((m - 1) * h * t) / (1 + v + (h * t * e))

    # Calculate the theoretical and practical correlation coefficients
    r = np.mean(np.multiply(x, y))
    r_th = np.sqrt(snr / (1 + snr))

    # The theoretical and practical correlation coefficients must be somewhat close
    np.testing.assert_almost_equal(r, r_th, decimal=1)
    return snr, r, r_th


def frame_error_rate_calculation(n_succ, n_tot):

    p_EC = n_succ / n_tot
    FER = 1 - p_EC
    if FER == 1.0:
        raise RuntimeWarning("There was no frame that was successfully decoded. A secret key cannot be formed.")
    return p_EC, FER


def verification(mA, mB, b, fnd):

    if fnd:  # If the syndromes are equal after error correction
        # If the hash values of the codewords are equal, verification is successful
        if hashes.universal_hashing(mA, mB, b):
            return True
        else:  # If the hash values do not match, verification has failed
            return False
    else:  # If the syndromes are not equal after error correction, the frame is automatically discarded
        return False

def composable_key_rate(n_bks, N, n, p, q, r_code, r_m_star, h_m, p_ec, e_sm, e_hash, e_cor, e_pe, h_k):

    Delta_AEP = 4 * np.log2(2 ** (1 + p / 2) + 1) * np.sqrt(np.log2(18 / ((p_ec ** 2) * (e_sm ** 4))))
    Theta = np.log2(p_ec * (1 - ((e_sm ** 2) / 3))) + 2 * np.log2(np.sqrt(2) * e_hash)

    # The practical secret key rate can be compared with a corresponding theoretical rate
    r_tilde_star = r_m_star - (Delta_AEP / np.sqrt(n)) + (Theta / n)
    r_theo = ((n * p_ec) / N) * r_tilde_star

    # Practical composable key rate
    r_m = h_k + r_code * q - p - h_m
    r_tilde = r_m - (Delta_AEP / np.sqrt(n)) + (Theta / n)  # The composable key rate without finite-size effects
    r_final = ((n * p_ec) / N) * r_tilde  #The composable key rate under finite-size effects

    # Privacy amplification input
    n_pa = int(p_ec * n_bks * p * n)  # The bit length of the concatenated decoded sequences
    r = int(np.ceil(p_ec * n_bks * n * r_tilde))  # The length of the final key

    # The ε-security of the protocol
    e = e_cor + e_sm + e_hash + p_ec * e_pe

    # Ensure the composable key rate is positive. There may be a case of the practical key rate being positive, while
    # the theoretical key rate is negative. This is communicated to the user for future simulation reference.
    if r_final < 0:
        warnings.warn("The composable key rate is negative. Privacy amplification cannot be performed.")
    elif r_theo < 0:
        warnings.warn("The theoretical composable key rate is negative. This means that future simulations under the"
                      "specified parameters will highly likely return a negative composable key rate.")

    return r_final, r_theo, n_pa, r, e
