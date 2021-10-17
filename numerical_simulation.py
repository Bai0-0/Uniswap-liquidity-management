import numpy as NP
import pandas as PD
NP.random.seed(1234)
import v3_core as VC
import cal_key_paras as KP

import os
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

class UsedPriceSeries:

    def __init__(self, price_series: NP.array):
        self.price_series = price_series
        self.current_location =  0
        self.current_price = Dec(price_series[self.current_location])

    def generate_next_price(self) -> Dec:
        self.current_location += 1
        self.current_price = Dec(self.price_series[self.current_location])
        return Dec(self.current_price)

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
    # test_price_series = PriceSeries(0, 4e-3)
    # PD.DataFrame({
    #     'price': [test_price_series.generate_next_price()[0] for _ in range(int(200000))]
    # }).to_csv('simulation_price.csv')
    store_path = 'numerical_detail/'
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    # test_price_series = PriceSeries(0, 4e-3)

    # price_records = [float(test_price_series.generate_next_price()[0]) for _ in range(5000)]
    price_records = PD.read_csv("simulation_price.csv")['price'][:5000]
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
        used_price_series = UsedPriceSeries(PD.read_csv('simulation_price.csv')['price'].values)

        initial_price = used_price_series.current_price
        print('Processing b', b)
        deposit_alpha = Dec(10.)
        test_user = {
            'strategy': Strategy(initial_price * NP.exp(-b), initial_price * NP.exp(b)),
            'reserves': VC.TokenInfo(deposit_alpha, deposit_alpha / initial_price),
            'range_order': VC.RangeOrder(initial_price, initial_price * Dec.exp(-a), initial_price * Dec.exp(a))
        }
        test_user['range_order'].fix_alpha(test_user['reserves'].alpha)
        # print(f'liquidity {range_order.liquidity} virtual alpha {range_order.virtual_alpha} virtual beta {range_order.virtual_beta}')
        range_order = test_user['range_order']
        initial_reserve = {
            'alpha': range_order.reserve.alpha,
            'beta': range_order.reserve.beta
        }
        # print(initial_reserve)
        print()

        rebase_cost = Dec(10)

        price_records, swap_records, reserve_alpha, reserve_beta, fee_alpha, fee_beta, token_wealth, fee_wealth = [initial_price], [''], [initial_reserve['alpha']], [initial_reserve['beta']], [Dec(0.)], [Dec(0.)], [initial_reserve['alpha'] + initial_reserve['beta'] * initial_price], [Dec(0.)]
        epoc_swap_begin_index, epoc_invest_price, epoc_invest_price_high, epoc_invest_price_low, epoc_invest_alpha, epoc_invest_beta, epoc_swap_end_index, epoc_end_price, epoc_end_alpha_pool, epoc_end_beta_pool, epoc_end_alpha_fee, epoc_end_beta_fee, epoc_begin_wealth, epoc_end_wealth = [Dec(0)], [initial_price], [test_user['range_order'].price_high], [test_user['range_order'].price_low], [initial_reserve['alpha']], [initial_reserve['beta']], [], [], [], [], [], [], [test_user['range_order'].wealth], []
        effective_tx_count_ls = []
        effective_tx_count = 0

        total_swap_times = 150000
        for swap_times in range(total_swap_times):
            # update current price to range order
            test_user['range_order'].current_price = used_price_series.current_price
            old_price = test_user['range_order'].current_price
            price_change_info = used_price_series.generate_next_price()
            new_price = used_price_series.current_price

            one_swap = test_user['range_order'].cal_swap(old_price, new_price)
            if one_swap.alpha == None:
                swap_info = f'beta {one_swap.beta}'
            else:
                swap_info = f'alpha {one_swap.alpha}'
            price_records.append(used_price_series.current_price)
            if test_user['strategy'].signal(used_price_series.current_price):
                test_user['range_order'].meet_swap(one_swap)
                # print(f'\rSwap {swap_info} and Reserve is {test_user["range_order"].reserve}', end='')
                swap_records.append(swap_info)
                reserve_alpha.append(test_user["range_order"].reserve.alpha)
                reserve_beta.append(test_user["range_order"].reserve.beta)
                fee_alpha.append(test_user['range_order'].transaction_fee.alpha)
                fee_beta.append(test_user['range_order'].transaction_fee.beta)
                token_wealth.append(test_user['range_order'].wealth)
                fee_wealth.append(test_user['range_order'].transaction_fee.alpha + used_price_series.current_price * test_user['range_order'].transaction_fee.beta)
                if (fee_alpha[-1] - fee_alpha[-2]) > 0.00001 or (fee_beta[-1] - fee_beta[-2]) > 0.00001:
                    effective_tx_count += 1
            else:
                swap_records.append('rebase')

                print(f'\nRebase! Swap time {swap_times}\n------------------\n')

                epoc_swap_end_index.append(swap_times)
                epoc_end_alpha_pool.append(test_user["range_order"].reserve.alpha)
                epoc_end_alpha_fee.append(test_user['range_order'].transaction_fee.alpha)
                epoc_end_beta_pool.append(test_user["range_order"].reserve.beta)
                epoc_end_beta_fee.append(test_user['range_order'].transaction_fee.beta)
                epoc_end_price.append(used_price_series.current_price)
                effective_tx_count_ls.append(effective_tx_count)
                effective_tx_count = 0

                initial_price = used_price_series.current_price
                test_user = {
                    'strategy': Strategy(initial_price * Dec.exp(-b), initial_price * Dec.exp(b)),
                    'reserves': VC.TokenInfo(deposit_alpha, deposit_alpha / initial_price),
                    'range_order': VC.RangeOrder(initial_price, initial_price * Dec.exp(-a), initial_price * Dec.exp(a))
                }
                test_user['range_order'].fix_alpha(test_user['reserves'].alpha)
                range_order = test_user['range_order']
                initial_reserve = {
                    'alpha': range_order.reserve.alpha,
                    'beta': range_order.reserve.beta
                }
                epoc_swap_begin_index.append(swap_times + 1)
                epoc_invest_alpha.append(test_user["range_order"].reserve.alpha)
                epoc_invest_beta.append(test_user['range_order'].reserve.beta)
                epoc_invest_price.append(used_price_series.current_price)
                epoc_invest_price_high.append(test_user['range_order'].price_high)
                epoc_invest_price_low.append(test_user['range_order'].price_low)


                reserve_alpha.append(test_user["range_order"].reserve.alpha)
                reserve_beta.append(test_user["range_order"].reserve.beta)
                fee_alpha.append(test_user['range_order'].transaction_fee.alpha)
                fee_beta.append(test_user['range_order'].transaction_fee.beta)
                token_wealth.append(test_user['range_order'].wealth)
                fee_wealth.append(test_user['range_order'].transaction_fee.alpha + used_price_series.current_price * test_user['range_order'].transaction_fee.beta)

        epoc_swap_end_index.append(total_swap_times + 1)
        epoc_end_alpha_pool.append(test_user["range_order"].reserve.alpha)
        epoc_end_alpha_fee.append(test_user['range_order'].transaction_fee.alpha)
        epoc_end_beta_pool.append(test_user["range_order"].reserve.beta)
        epoc_end_beta_fee.append(test_user['range_order'].transaction_fee.beta)
        epoc_end_price.append(used_price_series.current_price)
        effective_tx_count_ls.append(effective_tx_count)


        transaction_records = PD.DataFrame({
            'swap_index': [_ for _ in range(len(price_records))],
            'current_price': price_records,
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
            'epoc_end_price': epoc_end_price,
            'epoc_effective_tx_count': effective_tx_count_ls
        })
        epoc_records['invest_wealth'] = epoc_records['epoc_invest_alpha'] + epoc_records['epoc_invest_beta'] * epoc_records['epoc_invest_price']
        epoc_records['end_wealth_in_pool'] = epoc_records['epoc_end_alpha_pool'] + epoc_records['epoc_end_beta_pool'] * epoc_records['epoc_end_price']
        epoc_records['end_wealth_fee'] = epoc_records['epoc_end_alpha_fee'] + epoc_records['epoc_end_beta_fee'] * epoc_records['epoc_end_price']
        epoc_records['end_wealth'] = epoc_records['end_wealth_in_pool'] + epoc_records['end_wealth_fee']
        epoc_records['holding_wealth'] = epoc_records['epoc_invest_alpha'] + epoc_records['epoc_invest_beta'] * epoc_records['epoc_end_price']

        # break
        transaction_records.to_csv(f'{store_path}{str(round(float(b),2)).replace(".", "_")}_transaction.csv')
        epoc_records.to_csv(f'{store_path}{str(round(float(b),2)).replace(".", "_")}_epoc.csv')