import sympy

import numpy as NP
import pandas as PD
NP.random.seed(123)
import v3_core as VC
import cal_key_paras as KP

from decimal import Decimal as Dec

class PriceSeries:

    def __init__(self, change_mean, change_variance):
        self.change_mean = change_mean # required to be zero under math
        self.change_variance = change_variance # for change variance, the smaller, the better

        self.initial_price = Dec(NP.exp(NP.random.normal(change_mean, change_variance)))
        self.current_price = self.initial_price

    def generate_next_price(self) -> tuple:
        random_direction = Dec(NP.random.normal(self.change_mean, self.change_variance))
        self.current_price *= Dec.exp(random_direction)
        return self.current_price, random_direction

class SwapSeries:
    def __init__(self):
        pass

    def generate_nex_swap(self, indicator, range_order, new_price) -> VC.Swap:
        '''
        # generate next swap based on range information and price change, if the price does not go out of range of the order, then swap random selected amount of alpha(uniform(0, 0.9) * total alpha) or beta, otherwise extract total amount of specified coin
        :param indicator: price go up(if indicator > 0, i.e withdraw beta)
        :param range_order: range order information
        :param new_price: next price
        :return: Swap
        '''
        if new_price >= range_order.price_high:
            return VC.Swap(beta=Dec(NP.infty))
        elif new_price <= range_order.price_low:
            return VC.Swap(alpha=Dec(NP.infty))
        proportion = Dec(NP.random.uniform(0.2, 0.3))
        if indicator > 0:
            return VC.Swap(beta=proportion * Dec(10.))
        return VC.Swap(alpha=proportion * Dec(10.))
        # if new_price <= range_order.price_low or new_price >= range_order.price_high:
        #     return VC.Swap(alpha=0.)
        #
        # temp_beta = range_order.reserve.beta - (NP.sqrt(float(range_order.liquidity ** 2 * new_price)) - range_order.liquidity * NP.sqrt(range_order.price_low))
        # temp_alpha = range_order.reserve.alpha - (NP.sqrt(float(range_order.liquidity ** 2 / new_price)) - range_order.liquidity / NP.sqrt(range_order.price_high))
        #
        # if temp_beta > 0:
        #     return VC.Swap(beta=temp_beta)
        # else:
        #     return VC.Swap(alpha=temp_alpha)
        # old_price = (range_order.reserve.alpha + range_order.liquidity / NP.sqrt(range_order.price_high)) / (range_order.reserve.beta + range_order.liquidity * NP.sqrt(range_order.price_low))
        # if new_price > old_price:
        #     # solution = sympy.solve(f'({range_order.reserve.alpha} + x * {new_price} + {range_order.liquidity} / {NP.sqrt(range_order.price_high)}) * ({range_order.reserve.beta} - x + {range_order.liquidity} * {NP.sqrt(range_order.price_low)}) - {range_order.liquidity} ** 2')
        #     # # solution = sympy.solve(f'({range_order.reserve.beta} - x + {range_order.liquidity} * {NP.sqrt(range_order.price_low)}) ** 2 / {new_price} - {range_order.liquidity} ** 2')
        #     #
        #     # print(f'solution {solution} alpha {solution[-1] * new_price} beta {solution[-1]}')
        #     virtual_beta = NP.sqrt(float(range_order.liquidity ** 2 * new_price))
        #     new_swap = VC.Swap(beta=range_order.reserve.beta - (virtual_beta - range_order.liquidity * NP.sqrt(range_order.price_low)))
        # else:
        #     # solution = sympy.solve(f'({range_order.reserve.alpha} - x + {range_order.liquidity} / {NP.sqrt(range_order.price_high)}) * ({range_order.reserve.beta} - x / {new_price} + {range_order.liquidity} * {NP.sqrt(range_order.price_low)}) - {range_order.liquidity} ** 2')
        #     # # solution = sympy.solve(f'({range_order.reserve.alpha} - x + {range_order.liquidity} / {NP.sqrt(range_order.price_high)}) ** 2 * {new_price} - {range_order.liquidity} ** 2')
        #     # print(f'solution {solution} alpha {solution[-1]} beta {solution[-1] / new_price}')
        #     virtual_alpha = NP.sqrt(float(range_order.liquidity ** 2 / new_price))
        #     new_swap = VC.Swap(alpha=range_order.reserve.alpha - (virtual_alpha - range_order.liquidity / NP.sqrt(range_order.price_high)))
        # return new_swap

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

if __name__ == '__main__':
    test_price_series = PriceSeries(0, 4e-3)
    test_swap_series = SwapSeries()
    

    price_records = [float(test_price_series.generate_next_price()[0]) for _ in range(5000)]
    print(KP.cal_moment(price_records))
    print(KP.cal_psi(price_records))
    result_psi = KP.cal_psi(price_records)
    if result_psi > 0.5:
        result_b = NP.log(2) + NP.log(result_psi + NP.sqrt(result_psi ** 2 - 1 / 4))
        print(result_b)
    else:
        raise Exception
    a, b = Dec(0.01), Dec(result_b)
    # try:
    for b in [Dec(0.2), Dec(0.4), Dec(0.6), Dec(0.8), Dec(1.0), Dec(result_b)]:
        test_price_series.current_price = Dec(1.)
        initial_price = test_price_series.current_price
        print('Processing b', b)
        deposit_alpha = Dec(10.)
        test_user = {
            'strategy': Strategy(initial_price * NP.exp(-b), initial_price * NP.exp(b)),
            'reserves': VC.TokenInfo(deposit_alpha, Dec(0)),
            'range_order': VC.RangeOrder(initial_price, initial_price * Dec.exp(-a), initial_price * Dec.exp(a))
        }
        test_user['range_order'].fix_alpha(test_user['reserves'].alpha)
        range_order = test_user['range_order']
        new_price = range_order.current_price
        # print(f'liquidity {range_order.liquidity} virtual alpha {range_order.virtual_alpha} virtual beta {range_order.virtual_beta}')
        initial_reserve = {
            'alpha': test_user['range_order'].reserve.alpha,
            'beta': test_user['range_order'].reserve.beta
        }
        # print(initial_reserve)
        print()

        rebase_cost = Dec(10)

        price_records, swap_records, reserve_alpha, reserve_beta, fee_alpha, fee_beta, token_wealth, fee_wealth, signal_list = [initial_price], [''], [initial_reserve['alpha']], [initial_reserve['beta']], [Dec(0.)], [Dec(0.)], [initial_reserve['alpha'] + initial_reserve['beta'] * initial_price], [Dec(0.)], [0.]
        epoc_swap_begin_index, epoc_invest_price, epoc_invest_price_high, epoc_invest_price_low, epoc_invest_alpha, epoc_invest_beta, epoc_swap_end_index, epoc_end_price, epoc_end_alpha_pool, epoc_end_beta_pool, epoc_end_alpha_fee, epoc_end_beta_fee, epoc_begin_wealth, epoc_end_wealth = [Dec(0)], [initial_price], [test_user['range_order'].price_high], [test_user['range_order'].price_low], [initial_reserve['alpha']], [initial_reserve['beta']], [], [], [], [], [], [], [test_user['range_order'].wealth], []

        total_swap_times = 100000
        for swap_times in range(total_swap_times):
            # update current price to range order
            test_user['range_order'].current_price = test_price_series.current_price
            price_change_info = test_price_series.generate_next_price()
            signal_list.append(price_change_info[1])
            one_swap = test_swap_series.generate_nex_swap(price_change_info[1], test_user['range_order'], price_change_info[0])
            if one_swap.alpha == None:
                swap_info = f'beta {one_swap.beta}'
            else:
                swap_info = f'alpha {one_swap.alpha}'
            price_records.append(test_price_series.current_price)
            if test_user['strategy'].signal(test_price_series.current_price):
                test_user['range_order'].meet_swap(one_swap)
                # print(f'\rSwap {swap_info} and Reserve is {test_user["range_order"].reserve}', end='')
                swap_records.append(swap_info)
                reserve_alpha.append(test_user["range_order"].reserve.alpha)
                reserve_beta.append(test_user["range_order"].reserve.beta)
                fee_alpha.append(test_user['range_order'].transaction_fee.alpha)
                fee_beta.append(test_user['range_order'].transaction_fee.beta)
                token_wealth.append(test_user['range_order'].wealth)
                fee_wealth.append(test_user['range_order'].transaction_fee.alpha + test_price_series.current_price * test_user['range_order'].transaction_fee.beta)
            else:
                swap_records.append('rebase')

                print(f'\nRebase! Swap time {swap_times}\n------------------\n')

                epoc_swap_end_index.append(swap_times)
                epoc_end_alpha_pool.append(test_user["range_order"].reserve.alpha)
                epoc_end_alpha_fee.append(test_user['range_order'].transaction_fee.alpha)
                epoc_end_beta_pool.append(test_user["range_order"].reserve.beta)
                epoc_end_beta_fee.append(test_user['range_order'].transaction_fee.beta)
                epoc_end_price.append(test_price_series.current_price)

                initial_price = test_price_series.current_price
                test_user = {
                    'strategy': Strategy(initial_price * Dec.exp(-b), initial_price * Dec.exp(b)),
                    'reserves': VC.TokenInfo(deposit_alpha, Dec(0)),
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

        epoc_swap_end_index.append(total_swap_times + 1)
        epoc_end_alpha_pool.append(test_user["range_order"].reserve.alpha)
        epoc_end_alpha_fee.append(test_user['range_order'].transaction_fee.alpha)
        epoc_end_beta_pool.append(test_user["range_order"].reserve.beta)
        epoc_end_beta_fee.append(test_user['range_order'].transaction_fee.beta)
        epoc_end_price.append(test_price_series.current_price)


        transaction_records = PD.DataFrame({
            'swap_index': [_ for _ in range(len(price_records))],
            'current_price': price_records,
            'signal_info': signal_list,
            'single_swap': swap_records,
            'reserve_alpha': reserve_alpha,
            'reserve_beta': reserve_beta,
            'fee_alpha': fee_alpha,
            'fee_beta': fee_beta,
            'token_wealth': token_wealth,
            'fee_wealth': fee_wealth
        })
        transaction_records['wealth_hold'] = initial_reserve['alpha'] + initial_reserve['beta'] * transaction_records['current_price'].values
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
        transaction_records.to_csv(f'results/{str(round(float(b),2)).replace(".", "_")}_transaction.csv')
        epoc_records.to_csv(f'results/{str(round(float(b),2)).replace(".", "_")}_epoc.csv')

    # except:
    #     pass
    

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