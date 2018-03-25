import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from one_asset_derivatives_functions import generating_stock, BlackScholes, delta_func, generating_interest, precise_delta

risk_free = 0.015
r1 = risk_free
r2 = risk_free
q = 0
b1 = r1 - q
b2 = r2 - q
sigma1 = 0.2
S0 = 100
X = 100
time = 1 / 365
T_list = np.arange(181, -1, -1)
length = 182


def monte_carlo_simulation(S0, risk_free, sigma1, X, T_list, length, time):
    '''
    ##########################
    #####       주가      #####
    ##########################
    '''
    stocks = generating_stock(S0, risk_free, sigma1, time, length)
    # stocks = [49, 48.12, 47.37, 50.25, 51.75,
    #           53.12, 53, 51.87, 51.38, 53,
    #           49.88, 48.50, 49.88, 50.37, 52.13,
    #           51.88, 52.87, 54.87, 54.62,
    #           55.87, 57.25]
    '''
    ###############################
    #####       옵션, 델타      #####
    ###############################
    '''

    option_premium_array = np.zeros(length)
    delta1_array = np.zeros(length)

    for i in range(length):
        S1 = float(stocks[i])
        T = round((T_list[i] / 365), 11)
        # print('T: ', T)
        # print(S1)
        # print('##')
        # print(S2)

        # 0.8, 0.7
        # sigma1,  sigma2
        temp = BlackScholes('C', S1, X, risk_free, sigma1, T) * (-1)

        option_premium_array[i] = temp
        # delta1_array[i] = delta_func('C', S1, X, risk_free, sigma1, T)
        delta1_array[i] = precise_delta(S1, X, risk_free, sigma1, T)

    # print(type(option_premium_list))
    '''
    ###############################
    #####       주식보유수       #####
    ###############################
    '''
    tot_S1 = -delta1_array

    '''
    ###############################
    #####        헷지 수        #####
    ###############################
    '''
    # S1 헷지수
    hedge_number_S1 = np.zeros(length)
    hedge_number_S1[0] = delta1_array[0]
    hedge_number_S1[1:] = np.diff(delta1_array)
    hedge_number_S1 *= -1

    '''
    ###############################
    #####       평가 손익       #####
    ###############################
    '''
    # 보유 주식 평가 가치 변화
    value_change_S1 = np.zeros(length)
    value_change_S1[1:] = np.diff(stocks) * tot_S1[:-1]

    # 옵션의 가치 변화
    value_change_option = np.zeros(length)
    value_change_option[1:] = np.diff(option_premium_array)

    # 포트폴리오의 가치 변화
    value_change_portfolio = value_change_S1 + value_change_option

    '''
    ###############################
    ####   현금 흐름(실현 손익)    #####
    ###############################
    '''
    # 주가 현금흐름
    cashflow_S1 = stocks * (-hedge_number_S1)

    # 옵션 현금흐름
    cashflow_option = np.zeros(length)
    cashflow_option[0] = -option_premium_array[0]
    if option_premium_array[length - 1] != 0:
        cashflow_option[length - 1] = stocks[length - 1] + option_premium_array[length - 1]
    elif option_premium_array[length - 1] == 0:
        cashflow_option[length - 1] = 0

    # 포트폴리오 현금흐름
    cashflow_portfolio = cashflow_S1 + cashflow_option
    # cashflow_portfolio = cashflow_S1
    '''
    ###############################
    #####       이자 비용       #####
    ###############################
    '''
    # 이자포함 누적 손익
    cumulative_cost_including_interest = np.zeros(length)
    cumulative_cost_including_interest[0] = cashflow_portfolio[0]

    # 이자 비용
    interest_cost = np.zeros(length)
    interest_cost[0] = generating_interest(cumulative_cost_including_interest[0], risk_free, time)

    for i in range(length - 1):
        cumulative_cost_including_interest[i + 1] = cashflow_portfolio[i + 1] + cumulative_cost_including_interest[i] + \
                                                    interest_cost[i]
        if i == length - 2:
            interest_cost[i + 1] = 0
        else:
            interest_cost[i + 1] = generating_interest(cumulative_cost_including_interest[i + 1], risk_free, time)

    # in the money로 마감했을 경우
    # 사고 팔았던 주식 정리 매매 해줘야 한다.

    if delta1_array[length - 1] == 1:
        cumulative_cost_including_interest[length - 1] \
            = cumulative_cost_including_interest[length - 1] - delta1_array[length - 1] * stocks[length - 1]
    else:
        None

    result = {
        'T_list': T_list,
        'option_premium_array': option_premium_array,
        'stocks_1': stocks,
        'delta1_array': delta1_array,
        'hedge_number_S1': hedge_number_S1,
        'cashflow_S1': cashflow_S1,
        'cashflow_option': cashflow_option,
        'cashflow_portfolio': cashflow_portfolio,
        'cumulative_cost_including_interest': cumulative_cost_including_interest,
        'interest_cost': interest_cost,
        'value_change_S1': value_change_S1,
        'value_change_option': value_change_option,
        'value_change_portfolio': value_change_portfolio
    }
    return result


'''
###############################
#####       시뮬레이션       #####
###############################
'''
simulation_number = 50
simulation_results = np.zeros((simulation_number, 2))
stocks_1_temp = np.zeros((simulation_number, length))



for i in range(simulation_number):
    result_ex1 = monte_carlo_simulation(S0, risk_free, sigma1, X, T_list, length, time)
    result_option_at_maturity = \
        result_ex1['option_premium_array'][length - 1]
    result_total_cashflow = \
        result_ex1['cumulative_cost_including_interest'][length - 1]

    simulation_results[i][0] = result_option_at_maturity
    simulation_results[i][1] = result_total_cashflow

    print('simulation times: {}'.format(i))
    '''
    모든 시뮬레이션을 엑셀로 저장하고 싶으면 아래의 코드를 활성화 시키고 진행하면 된다.
    '''
    # T_list_r = result_ex1['T_list']
    # option_premium_array_r = result_ex1['option_premium_array']
    # stocks_1_r = result_ex1['stocks_1']
    #
    # stocks_1_temp[i] = result_ex1['stocks_1']
    #
    # simulation_results[i][0] = result_ex1['option_premium_array'][length - 1]
    # simulation_results[i][1] = result_ex1['cumulative_cost_including_interest'][length - 1]
    #
    # delta1_array_r = result_ex1['delta1_array']
    # hedge_number_S1_r = result_ex1['hedge_number_S1']
    # cashflow_S1_r = result_ex1['cashflow_S1']
    # cashflow_option_r = result_ex1['cashflow_option']
    # cashflow_portfolio_r = result_ex1['cashflow_portfolio']
    # cumulative_cost_including_interest_r = result_ex1['cumulative_cost_including_interest']
    # interest_cost_r = result_ex1['interest_cost']
    # value_change_S1_r = result_ex1['value_change_S1']
    # value_change_option_r = result_ex1['value_change_option']
    # value_change_portfolio_r = result_ex1['value_change_portfolio']
    # empty_column = np.zeros(length)
    #
    # values1 = list(
    #     zip(T_list_r, stocks_1_r, option_premium_array_r, delta1_array_r, hedge_number_S1_r, cashflow_S1_r,
    #         cashflow_option_r, cashflow_portfolio_r, cumulative_cost_including_interest_r,
    #         interest_cost_r, empty_column,
    #         value_change_S1_r, value_change_option_r, value_change_portfolio_r))
    #
    # df = pd.DataFrame(values1,
    #                   columns=['잔존만기', 'S1', '옵션가격', 'delta_S1', 'S1헷지수', '주식1현금흐름',
    #
    #                            '옵션 현금흐름', '포트폴리오 현금흐름', '이자포함 누적손익', '이자', '----------', 'S1 주가 평가 가치 변화',
    #
    #                            '옵션 평가 가치 변화', '포트폴리오 평가 가치 변화'])
    # print('######################')
    # print(delta1_array_r[length - 1], stocks_1_r[length - 1])
    # print('######################')
    # df.to_excel("result{}".format(i) + '.xls', encoding='utf-8')
# state = np.array([['옵션 payoff', '누적 손익']])
non_zero_count = np.count_nonzero(simulation_results[:, 0])
avg_of_total_cashflow = np.mean(simulation_results[:, 1])
# print(state)
# print(simulation_results)
# print('시뮬레이션 결과: \n', np.concatenate((state, simulation_results), axis=0))
print('{}번의 시행중 옵션 payoff > 0 인 횟수: '.format(simulation_number), non_zero_count)
print('옵션을 행사할 확률: ', non_zero_count / simulation_number * 100, '%')
print('델타헷지의 이자포함누적손익 평균: ', avg_of_total_cashflow)
values = simulation_results
df = pd.DataFrame(values, columns=['옵션 payoff', '누적 손익'])
writer = pd.ExcelWriter('one_asset_result_of_monte_carlo_simulation.xlsx')
df.to_excel(writer, encoding='utf-8')
writer.save()
plt.hist(simulation_results[:, 1], bins=100)
#
# max_ind = np.argmax(simulation_results[:, 1])
# max_s1 = stocks_1_temp[max_ind]
#
# min_ind = np.argmin(simulation_results[:, 1])
# min_s1 = stocks_1_temp[min_ind]
#
# plt.figure(figsize=(8, 6))
# plt.plot(max_s1, c='orange')
# plt.xlabel(simulation_results[:, 1][max_ind])
# plt.title('MAX')
#
# plt.figure(figsize=(8, 6))
# plt.plot(min_s1, c='orange')
# plt.xlabel(simulation_results[:, 1][min_ind])
# plt.title('MIN')
# print('최대값이 나오는 숫자:', max_ind, '최소값이 나오는 숫자', min_ind)
plt.show()
