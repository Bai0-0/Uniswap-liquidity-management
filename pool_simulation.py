import sympy, os
import numpy as NP
import pandas as PD
from namedlist import namedlist
from decimal import Decimal as Dec

import v3_core as VC
from typing import Union, Any

TokenInfo = namedlist('Token', ['alpha', 'beta'])

class PriceCache(list):
    def __init__(self, *args):
        super(PriceCache, self).__init__(*args)

    def update(self, new_price):
        self.append(float(new_price))
        if len(self) > 100:
            self.pop(0)

class PriceSeries:

    def __init__(self, price_series: NP.array):
        self.price_series = price_series
        self.current_location =  0
        self.current_price = price_series[self.current_location]

    def generate_next_price(self) -> Dec:
        self.current_location += 1
        self.current_price = self.price_series[self.current_location]
        return self.current_price

class Strategy:

    def __init__(self, break_point_low, break_point_high):
        '''
        key parameters to determine strategy
        :param break_point_low: lowest price to rebase
        :param break_point_high: highest price to rebase
        '''
        self.break_point_low = break_point_low
        self.break_point_high = break_point_high

    def signal(self, price) -> bool:
        if price > self.break_point_high or price < self.break_point_low:
            return False
        return True

def cal_moment(price_series):
    # price_series = price_records
    price_changes = NP.diff(NP.log(NP.array(price_series)))
    return NP.mean(NP.abs(price_changes)), NP.var(price_changes)

def cal_psi(price_series, rate=0.003):
    moments = cal_moment(price_series)
    return rate * 2 * moments[0] / moments[1]

def initialize_pool_info(pool_info_path: str) -> PriceSeries:
    '''
    initialize raw pool information into price series and swap series
    :param pool_info_path: file path for liquidity pool information
    :return: PriceSeries and SwapSeries
    '''
    # pool_info_path = 'LINK_WETH_3000_0815.csv'
    pool_info = PD.read_csv(f'../data/after_preprocessing/{pool_info_path}')
    swap_info = pool_info.loc[pool_info['event_name'] == 'Swap'].reset_index(drop=True)
    swap_info = swap_info.drop(index=[0, 1]).reset_index(drop=True)
    abstract_info = swap_info[['liquidity', 'amount0_delta', 'amount1_delta', 'p_real']]
    price_series = PriceSeries(price_series=NP.array([Dec(1.) / Dec(_) for _ in abstract_info['p_real'].values]))
    return price_series
    # initialize swap series
    # swap_list = []
    # for swap_index in range(len(abstract_info)):
    #     temp_liquidity = Dec(abstract_info['liquidity'].values[swap_index])
    #     new_price = Dec(abstract_info['p_real'].values[swap_index])
    #     delta_alpha = Dec(abstract_info['amount0_delta'].values[swap_index])
    #     delta_beta = Dec(abstract_info['amount1_delta'].values[swap_index])
    #     if delta_alpha > 0:
    #         temp_swap = Swap(beta=-delta_beta, liquidity=temp_liquidity, new_price=new_price)
    #     elif delta_beta > 0:
    #         temp_swap = Swap(alpha=-delta_alpha, liquidity=temp_liquidity, new_price=new_price)
    #     else:
    #         raise Exception
    #     swap_list.append(temp_swap)
    # return price_series, SwapSeries(swap_series=NP.array(swap_list))

def generate_transaction_records(pool_info_path, a: Dec, default_b: Dec=None, deposit_alpha=Dec(100.), steps=None):
    # pool_info_path = single_pool_info
    fee_rate = int(pool_info_path.split('_')[2]) / 1e6
    test_price_series = initialize_pool_info(pool_info_path)
    price_cache = PriceCache()
    if steps == None:
        steps = len(test_price_series.price_series) - 1
    initial_price = test_price_series.current_price
    price_cache.update(initial_price)
    test_user = {
        'reserves': TokenInfo(deposit_alpha, Dec(0)),
        'range_order': VC.RangeOrder(initial_price, initial_price * Dec.exp(-a), initial_price * Dec.exp(a))
    }
    test_user['range_order'].fix_alpha(test_user['reserves'].alpha)

    initial_reserve = {
        'alpha': test_user['range_order'].reserve.alpha,
        'beta': test_user['range_order'].reserve.beta
    }

    price_records, swap_records, reserve_alpha, reserve_beta, fee_alpha, fee_beta, token_wealth, fee_wealth, wealth_hold = [initial_price], [''], [initial_reserve['alpha']], [initial_reserve['beta']], [Dec(0.)], [Dec(0.)], [initial_reserve['alpha'] + initial_reserve['beta'] * initial_price], [Dec(0.)], [initial_reserve['alpha'] + initial_reserve['beta'] * initial_price]
    epoc_swap_begin_index, epoc_invest_price, epoc_invest_price_high, epoc_invest_price_low, epoc_invest_alpha, epoc_invest_beta, epoc_swap_end_index, epoc_end_price, epoc_end_alpha_pool, epoc_end_beta_pool, epoc_end_alpha_fee, epoc_end_beta_fee, epoc_begin_wealth, epoc_end_wealth = [Dec(0)], [initial_price], [test_user['range_order'].price_high], [test_user['range_order'].price_low], [initial_reserve['alpha']], [initial_reserve['beta']], [], [], [], [], [], [], [test_user['range_order'].wealth], []

    for swap_times in range(steps):
        # update current price to range order
        if len(price_cache) < 100:
            test_price_series.generate_next_price()
            price_cache.update(test_price_series.current_price)
            continue
        result_psi = cal_psi(price_cache, fee_rate)
        if default_b == None:
            if result_psi > 0.5:
                b = Dec(NP.log(2) + NP.log(result_psi + NP.sqrt(result_psi ** 2 - 1 / 4))) * Dec(0.5)
            else:
                # print(f'{result_psi} and b using default 0.1')
                b = Dec(0.1)
            print(f'Swap times {swap_times} Update b to be {round(float(b), 5)}')
            test_user['strategy'] = Strategy(initial_price * Dec.exp(-b), initial_price * Dec.exp(b))
        else:
            test_user['strategy'] = Strategy(initial_price * Dec.exp(-default_b), initial_price * Dec.exp(default_b))
        test_user['range_order'].current_price = test_price_series.current_price
        old_price = test_price_series.current_price
        test_price_series.generate_next_price()
        new_price = test_price_series.current_price
        one_swap = test_user['range_order'].cal_swap(old_price, new_price)
        price_cache.update(test_price_series.current_price)
        if one_swap.alpha == None:
            swap_info = f'beta {round(one_swap.beta, 2)}'
        else:
            swap_info = f'alpha {round(one_swap.alpha, 2)}'
        price_records.append(test_price_series.current_price)
        if test_user['strategy'].signal(test_price_series.current_price):
            test_user['range_order'].meet_swap(one_swap)
            # print(f'\rSwap {swap_info} and Reserve alpha as {round(test_user["range_order"].reserve.alpha, 2)} beta as {round(test_user["range_order"].reserve.beta, 2)}', end='')
            swap_records.append(swap_info)
            reserve_alpha.append(test_user["range_order"].reserve.alpha)
            reserve_beta.append(test_user["range_order"].reserve.beta)
            fee_alpha.append(test_user['range_order'].transaction_fee.alpha)
            fee_beta.append(test_user['range_order'].transaction_fee.beta)
            token_wealth.append(test_user['range_order'].wealth)
            fee_wealth.append(test_user['range_order'].transaction_fee.alpha + test_price_series.current_price * test_user['range_order'].transaction_fee.beta)
            wealth_hold.append(initial_reserve['alpha'] + initial_reserve['beta'] * test_price_series.current_price)
        else:
            swap_records.append('rebase')

            # print(f'Rebase! Swap time {swap_times}\n------------------\n')

            epoc_swap_end_index.append(swap_times)
            epoc_end_alpha_pool.append(test_user["range_order"].reserve.alpha)
            epoc_end_alpha_fee.append(test_user['range_order'].transaction_fee.alpha)
            epoc_end_beta_pool.append(test_user["range_order"].reserve.beta)
            epoc_end_beta_fee.append(test_user['range_order'].transaction_fee.beta)
            epoc_end_price.append(test_price_series.current_price)

            initial_price = test_price_series.current_price
            if default_b == None:
                assert 'b' in locals() or 'b' in globals()
                test_user = {
                    'strategy': Strategy(initial_price * Dec.exp(-b), initial_price * Dec.exp(b)),
                    'reserves': TokenInfo(deposit_alpha, Dec(0)),
                    'range_order': VC.RangeOrder(initial_price, initial_price * Dec.exp(-a), initial_price * Dec.exp(a))
                }
            else:
                test_user = {
                    'strategy': Strategy(initial_price * Dec.exp(-default_b), initial_price * Dec.exp(default_b)),
                    'reserves': TokenInfo(deposit_alpha, Dec(0)),
                    'range_order': VC.RangeOrder(initial_price, initial_price * Dec.exp(-a), initial_price * Dec.exp(a))
            }
            test_user['range_order'].fix_alpha(test_user['reserves'].alpha)
            initial_reserve = {
                'alpha': test_user['range_order'].reserve.alpha,
                'beta': test_user['range_order'].reserve.beta
            }
            epoc_swap_begin_index.append(swap_times + 1)
            epoc_invest_alpha.append(test_user["range_order"].reserve.alpha)
            epoc_invest_beta.append(test_user['range_order'].reserve.beta)
            epoc_invest_price.append(test_price_series.current_price)
            epoc_invest_price_high.append(test_user['range_order'].price_high)
            epoc_invest_price_low.append(test_user['range_order'].price_low)


            reserve_alpha.append(test_user["range_order"].reserve.alpha)
            reserve_beta.append(test_user["range_order"].reserve.beta)
            fee_alpha.append(test_user['range_order'].transaction_fee.alpha)
            fee_beta.append(test_user['range_order'].transaction_fee.beta)
            token_wealth.append(test_user['range_order'].wealth)
            fee_wealth.append(test_user['range_order'].transaction_fee.alpha + test_price_series.current_price * test_user['range_order'].transaction_fee.beta)
            wealth_hold.append(initial_reserve['alpha'] + initial_reserve['beta'] * test_price_series.current_price)

    epoc_swap_end_index.append(steps + 1)
    epoc_end_alpha_pool.append(test_user["range_order"].reserve.alpha)
    epoc_end_alpha_fee.append(test_user['range_order'].transaction_fee.alpha)
    epoc_end_beta_pool.append(test_user["range_order"].reserve.beta)
    epoc_end_beta_fee.append(test_user['range_order'].transaction_fee.beta)
    epoc_end_price.append(test_price_series.current_price)
    price_cache.update(test_price_series.current_price)

    transaction_records = PD.DataFrame({
        'swap_index': [_ for _ in range(len(price_records))],
        'current_price': price_records,
        'single_swap': swap_records,
        'reserve_alpha': reserve_alpha,
        'reserve_beta': reserve_beta,
        'fee_alpha': fee_alpha,
        'fee_beta': fee_beta,
        'token_wealth': token_wealth,
        'fee_wealth': fee_wealth,
        'wealth_hold': wealth_hold
    })
    # transaction_records['wealth_hold'] = initial_reserve['alpha'] + initial_reserve['beta'] * transaction_records['current_price'].values
    transaction_records['wealth_pool'] = transaction_records['token_wealth'] + transaction_records['fee_wealth']

    epoc_records = PD.DataFrame({
        'epoc_begin_index': epoc_swap_begin_index,
        'epoc_end_index': epoc_swap_end_index,
        'epoc_invest_alpha': epoc_invest_alpha,
        'epoc_invest_beta': epoc_invest_beta,
        'epoc_invest_price_high': epoc_invest_price_high,
        'epoc_invest_price_low': epoc_invest_price_low,
        'epoc_invest_price': epoc_invest_price,
        'epoc_end_alpha_pool': epoc_end_alpha_pool,
        'epoc_end_beta_pool': epoc_end_beta_pool,
        'epoc_end_alpha_fee': epoc_end_alpha_fee,
        'epoc_end_beta_fee': epoc_end_beta_fee,
        'epoc_end_price': epoc_end_price
    })
    epoc_records['invest_wealth'] = epoc_records['epoc_invest_alpha'] + epoc_records['epoc_invest_beta'] * epoc_records['epoc_invest_price']
    epoc_records['end_wealth_in_pool'] = epoc_records['epoc_end_alpha_pool'] + epoc_records['epoc_end_beta_pool'] * epoc_records['epoc_end_price']
    epoc_records['end_wealth_fee'] = epoc_records['epoc_end_alpha_fee'] + epoc_records['epoc_end_beta_fee'] * epoc_records['epoc_end_price']
    epoc_records['end_wealth'] = epoc_records['end_wealth_in_pool'] + epoc_records['end_wealth_fee']
    epoc_records['holding_wealth'] = epoc_records['epoc_invest_alpha'] + epoc_records['epoc_invest_beta'] * epoc_records['epoc_end_price']
    return transaction_records, epoc_records

total_pool_info = PD.read_csv("para_result.csv")
for single_index in range(len(total_pool_info)):
    # single_index = 9
    single_pool_info = total_pool_info.loc[single_index, 'Pool']
    print(f'Processing {single_index} {single_pool_info}')
    fee_rate = int(single_pool_info.split('_')[2]) / 1e6

    root_folder = 'pool_detail'
    a, b = Dec(0.01), Dec(0.15)
    folder_name = single_pool_info[: -4]
    saved_path = root_folder + '/' + folder_name
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    # transaction_records, epoc_records = generate_transaction_records(pool_info_path, a=a, default_b=Dec(0.15))
    for b_index in range(40):
        # print(f'processing {str(round((b_index + 1) * 0.05, 2))}')
        new_b = Dec((b_index + 1) * 0.05)
        file_name = str(round((b_index + 1) * 0.05, 2)).replace('.', '_')
        transaction_records, epoc_records = generate_transaction_records(single_pool_info, a=a, default_b=new_b)
        transaction_records.to_csv(f'{saved_path}/{file_name}_transaction.csv')
        epoc_records.to_csv(f'{saved_path}/{file_name}_epoc.csv')
    transaction_records, epoc_records = generate_transaction_records(single_pool_info, a=a)
    transaction_records.to_csv(f'{saved_path}/optimum_transaction.csv')
    epoc_records.to_csv(f'{saved_path}/optimum_epoc.csv')

# import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
# plt.plot(price_records)
# plt.show()
#
# x_axis = transaction_records['swap_index']
# plt.plot(x_axis, transaction_records['wealth_hold'].values, color='black', label='wealth in hand')
# plt.plot(x_axis, transaction_records['token_wealth'].values, color='red', label='token wealth in pool')
# plt.plot(x_axis, transaction_records['wealth_pool'].values, color='blue', label='wealth in pool')
# plt.legend()
# plt.show()