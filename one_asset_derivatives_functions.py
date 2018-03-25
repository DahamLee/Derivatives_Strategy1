# Calculate Input Parameters
import numpy as np
import scipy.stats as ss
import time


# Black and Scholes
def d1(S0, K, r, sigma, T):
    if T == 0:
        return None
    return (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def d2(S0, K, r, sigma, T):
    if T == 0:
        return None
    return (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def BlackScholes(type, S0, K, r, sigma, T):
    if type == "C":
        if T == 0:
            return max(S0 - K, 0)
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
        if T == 0:
            return max(K - S0, 0)
        return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))


def delta_func(type, S0, K, r, sigma, T):
    '''
    Calculate delta of c_min by S2
    '''
    # option_payoff = BlackScholes(type, S0, K, r, sigma, T)
    # if T == 0 and option_payoff > 0.95:
    #     return 1
    #
    # elif T == 0 and option_payoff < 0.05:
    #     return 0
    # else:
    #     ps1 = S0 + 0.0001
    #     ms1 = S0 - 0.0001
    #     pc1 = BlackScholes(type, ps1, K, r, sigma, T)
    #     mc1 = BlackScholes(type, ms1, K, r, sigma, T)
    #     delta1 = (pc1 - mc1) / 0.0002
    #     return delta1

    ps1 = S0 + 0.000001
    ms1 = S0 - 0.000001
    pc1 = BlackScholes(type, ps1, K, r, sigma, T) * (-1)
    mc1 = BlackScholes(type, ms1, K, r, sigma, T) * (-1)
    delta1 = (pc1 - mc1) / 0.000002
    return delta1


def recursion_fomula(initial_stock_price, mu, sigma, epsilon, time):
    return initial_stock_price * np.exp((mu - sigma ** 2 / 2) * time + sigma * epsilon * np.sqrt(time))


def generating_stock(initial_stock, mu1, sigma1, time, stock_numbers):
    s0 = initial_stock
    temp_list1 = list()

    for i in range(stock_numbers):
        epsilon1 = np.random.randn()
        if i == 0:
            s1 = s0

        else:
            s1 = recursion_fomula(s1, mu1, sigma1, epsilon1, time)
        temp_list1.append(s1)
    result = np.array((temp_list1))
    return result


def generating_interest(asset_value, riskfree_rate, time):
    return asset_value * (np.exp(riskfree_rate * time) - 1)


def precise_delta(S0, K, r, sigma, T):
    option_payoff = BlackScholes('C', S0, K, r, sigma, T) * (-1)
    if T == 0 and option_payoff < -0.00005:
        return -1

    elif T == 0 and option_payoff >= -0.00005:
        return 0
    else:
        a = d1(S0, K, r, sigma, T)
        return ss.norm.cdf(a) * (-1)
