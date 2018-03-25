import numpy as np
from scipy.integrate import dblquad
from scipy.stats import norm
from statsmodels.sandbox.distributions.extras import mvstdnormcdf

'''
###################################################################################################
################              Pricing Two Asset Option Function               #####################
###################################################################################################
'''


# Calculate Input Parameters
def calculate_params(S1, S2, X, T, b1, b2, sigma1, sigma2, rho):
    '''
    Calculate Input Parameters
    '''
    if T == 0:
        return None
    else:
        sigma = np.sqrt(sigma1 ** 2 + sigma2 ** 2 - 2 * rho * sigma1 * sigma2)

        rho1 = (sigma1 - rho * sigma2) / sigma
        rho2 = (sigma2 - rho * sigma1) / sigma

        d = (np.log(S1 / S2) + (b1 - b2 + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

        y1 = (np.log(S1 / X) + (b1 + sigma1 ** 2 / 2) * T) / (sigma1 * np.sqrt(T))
        y2 = (np.log(S2 / X) + (b2 + sigma2 ** 2 / 2) * T) / (sigma2 * np.sqrt(T))

        result = {'sigma': sigma, 'rho1': rho1, 'rho2': rho2, 'd': d, 'y1': y1, 'y2': y2}

        return result


# Call on the Minimum of Two Assets
def c_min(S1, S2, X, T, r, b1, b2, sigma1, sigma2, rho):
    '''
    Call on the Minimum of Two Assets
    '''
    if T == 0:
        return round(max(min(S1, S2) - X, 0), 11)
    else:
        calc = calculate_params(S1, S2, X, T, b1, b2, sigma1, sigma2, rho)
        sigma = calc['sigma']
        rho1 = calc['rho1']
        rho2 = calc['rho2']
        d = calc['d']
        y1 = calc['y1']
        y2 = calc['y2']

        return S1 * np.exp(b1 * T) * mvstdnormcdf([-np.inf, -np.inf], [y1, -d], -rho1) + \
               S2 * np.exp(b2 * T) * mvstdnormcdf([-np.inf, -np.inf], [y2, d - sigma * np.sqrt(T)], -rho2) - \
               X * np.exp(-r * T) * mvstdnormcdf([-np.inf, -np.inf],
                                                 [y1 - sigma1 * np.sqrt(T), y2 - sigma2 * np.sqrt(T)], rho)


'''
###################################################################################################
####################                Delta Function excel               ############################
###################################################################################################
'''


def delta_1_c_min_excel_version(S1, S2, X, T, r, b1, b2, sigma1, sigma2, rho):
    '''
    Calculate delta of c_min by S2
    '''
    option_payoff = c_min(S1, S2, X, T, r, b1, b2, sigma1, sigma2, rho)
    if T == 0 and option_payoff != 0:
        if S1 > S2:
            return 0
        else:
            return 1
    else:
        ps1 = S1 + 0.001
        ms1 = S1 - 0.001
        pc1 = c_min(ps1, S2, X, T, r, b1, b2, sigma1, sigma2, rho)
        # print(pc1)
        mc1 = c_min(ms1, S2, X, T, r, b1, b2, sigma1, sigma2, rho)
        # print(mc1)
        delta1 = (pc1 - mc1) / 0.002
        return delta1


def delta_2_c_min_excel_version(S1, S2, X, T, r, b1, b2, sigma1, sigma2, rho):
    '''
    Calculate delta of c_min by S2
    '''
    option_payoff = c_min(S1, S2, X, T, r, b1, b2, sigma1, sigma2, rho)
    if T == 0 and option_payoff != 0:
        if S1 < S2:
            return 0
        else:
            return 1
    else:
        ps2 = S2 + 0.001
        ms2 = S2 - 0.001
        pc2 = c_min(S1, ps2, X, T, r, b1, b2, sigma1, sigma2, rho)
        mc2 = c_min(S1, ms2, X, T, r, b1, b2, sigma1, sigma2, rho)
        delta2 = (pc2 - mc2) / 0.002
        return delta2


'''
###############################################################################
######################         stock generating         #######################
###############################################################################
'''


def recursion_fomula(initial_stock_price, mu, sigma, epsilon, time):
    return initial_stock_price * np.exp((mu - sigma ** 2 / 2) * time + sigma * epsilon * np.sqrt(time))


def generating_2stocks(initial_2stock_price_list, mu1, mu2, sigma1, sigma2, time, rho, stock_numbers):
    s0s = initial_2stock_price_list
    temp_list1 = list()
    temp_list2 = list()

    for i in range(stock_numbers):
        # epsilon = np.random.randn(2)
        # print(epsilon)
        epsilon1 = np.random.randn()
        epsilon2 = rho * epsilon1 + np.random.randn() * np.sqrt(1 - rho ** 2)
        # epsilon1 = epsilon[0]
        # epsilon2 = epsilon[1]
        if i == 0:
            try:
                s1 = s0s[0]
                s2 = s0s[1]
            except:
                print('** initial_2stock_price_list should be list type **')

        else:
            s1 = recursion_fomula(s1, mu1, sigma1, epsilon1, time)
            s2 = recursion_fomula(s2, mu2, sigma2, epsilon2, time)
        temp_list1.append(s1)
        temp_list2.append(s2)
    result = np.array((temp_list1, temp_list2))
    return result


def generating_logarithm_expected_return(array_of_stock):
    expected_return_s1 = [np.log(array_of_stock[0][i + 1] / array_of_stock[0][i]) for i in
                          range(len(array_of_stock[0]) - 1)]
    expected_return_s2 = [np.log(array_of_stock[1][i + 1] / array_of_stock[1][i]) for i in
                          range(len(array_of_stock[1]) - 1)]
    result = np.array((expected_return_s1, expected_return_s2))
    return result


'''
###############################################################################
###################          interest cost function          ##################
###############################################################################
'''


def generating_interest(asset_value, riskfree_rate, time):
    return asset_value * (np.exp(riskfree_rate * time) - 1)

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

# print('c_min: ', c_min(S1, S2, X, T, r, b1, b2, sigma1, sigma2, rho))
# print('delta_1_c_min_excel_version: ', delta_1_c_min_excel_version(S1, S2, X, T, r, b1, b2, sigma1, sigma2, rho))
# print('delta_2_c_min_excel_version: ', delta_2_c_min_excel_version(S1, S2, X, T, r, b1, b2, sigma1, sigma2, rho))
