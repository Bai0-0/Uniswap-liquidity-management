import sympy
import numpy as NP
from namedlist import namedlist
from decimal import Decimal as Dec
from typing import Union, Any
TokenInfo = namedlist('Token', ['alpha', 'beta'])

class Swap:
    '''swap in liquidity pool'''

    def __init__(self, alpha: Union[None, Dec]=None, beta: Union[None, Dec]=None):
        '''
        swap happened in pool
        :param alpha: amount of alpha to withdraw from pool, None indicating to withdraw beta
        :param beta: amount of beta to withdraw from pool, None indicating to withdraw alpha
        '''
        assert not ((alpha == None) and (beta == None)), 'Both alpha and beta are None, invalid swap'
        self.alpha = alpha
        self.beta = beta


class RangeOrder:
    '''range order placed into v3 pool'''

    def __init__(self, current_price: Dec, price_low: Dec, price_high: Dec, fee_rate=Dec(.003)):
        '''
        only range with current_price inside the interval
        :param current_price: current_price of uniswap pool
        :param price_low: left side of the interval, price_low < current_price
        :param price_high: right side of the interval, price_high > current_price
        :param fee_rate: fee rate for transaction
        '''
        assert (current_price > price_low) and (current_price < price_high), 'Invalid range interval'
        self.current_price = current_price
        self.price_low = price_low
        self.price_high = price_high
        self.fee_rate = fee_rate
        self.reserve = TokenInfo(Dec(0.), Dec(0.))
        self.initial_wealth, self.wealth = Dec(0.), Dec(0.)
        self.transaction_fee = TokenInfo(Dec(0.), Dec(0.))

    def fix_alpha(self, alpha: Dec) -> TokenInfo:
        '''
        fix alpha to determine how many beta to put into pool
        :param alpha: amount of alpha to put
        :return: alpha and beta entitled to the range order, and values return in order
        '''
        # (alpha + L / (sqrt(price + high))) * (beta + L * sqrt(price_low)) = L^2
        beta = (Dec(1) / Dec.sqrt(self.current_price) - Dec(1) / Dec.sqrt(self.price_high)) / (Dec.sqrt(self.current_price) - Dec.sqrt(self.price_low)) * alpha
        self.reserve.alpha = alpha
        self.reserve.beta = beta
        self.update_wealth()
        self.cal_liquidity()
        return self.reserve

    def fix_beta(self, beta: Dec) -> TokenInfo:
        '''
        fix beta to determine how many alpha to put into pool
        :param beta: amount of beta to put
        :return: alpha and beta entitled to the range order, and values return in order
        '''
        # (alpha + L / (sqrt(price + high))) * (beta + L * sqrt(price_low)) = L^2
        alpha = (Dec.sqrt(self.current_price) - Dec.sqrt(self.price_low)) / (Dec(1) / Dec.sqrt(self.current_price) - Dec(1) / Dec.sqrt(self.price_high)) * beta
        self.reserve.alpha = alpha
        self.reserve.beta = beta
        self.update_wealth()
        self.cal_liquidity()
        return self.reserve

    def meet_swap(self, swap: Swap) -> tuple:
        '''
        things to do if swap happens in this order
        :param swap: swap information
        :return: left alpha-beta in range order
        '''
        if swap.alpha == None:  # now the swap is going to withdraw beta out
            beta_to_withdraw = min(swap.beta, self.reserve.beta)
            self.reserve.beta -= beta_to_withdraw
            self.reserve.alpha = min(self.liquidity * self.liquidity / (self.reserve.beta + self.liquidity * Dec.sqrt(self.price_low)) - self.liquidity / Dec.sqrt(self.price_high), Dec(21))
            self.transaction_fee.beta += beta_to_withdraw * self.fee_rate
        elif swap.beta == None:  # the swap is going to withdraw alpha out
            alpha_to_withdraw = min(swap.alpha, self.reserve.alpha)
            self.reserve.alpha -= alpha_to_withdraw
            self.reserve.beta = min(self.liquidity * self.liquidity / (
                        self.reserve.alpha + self.liquidity / Dec.sqrt(self.price_high)) - self.liquidity * Dec.sqrt(self.price_low), 21)
            self.transaction_fee.alpha += alpha_to_withdraw * self.fee_rate
        self.update_wealth()
        return self.reserve

    def update_wealth(self) -> Dec:
        self.wealth = self.reserve.alpha + self.reserve.beta * self.current_price
        # print(self.wealth)
        return self.wealth

    def cal_liquidity(self) -> float:
        '''
        calculate liquidity given alpha, beta, price_low, price_high
        :return: update liquidity automatically, also return it back
        '''
        total_results = sympy.solve(
            f'({self.reserve.alpha} + x / {NP.sqrt(self.price_high)}) * ({self.reserve.beta} + x * {NP.sqrt(self.price_low)}) - x ** 2')
        # print(total_results)
        assert (total_results[0] < 0) or (len(total_results) == 1), str(
            total_results) + 'invalid results for virtual liquidity calculation'
        self.liquidity = total_results[-1]
        self.virtual_alpha = self.reserve.alpha + self.liquidity / NP.sqrt(self.price_high)
        self.virtual_beta = self.reserve.beta + self.liquidity * NP.sqrt(self.price_low)
        return self.liquidity


if __name__ == '__main__':
    a = Dec(0.1)
    test_order = RangeOrder(Dec(1), Dec.exp(-a), Dec.exp(a))
    test_order.fix_alpha(Dec(1000.))
    print('Initial state, reserve is', test_order.reserve)
    print('For current order, liquidity is', test_order.liquidity)
    print(f'Virtual alpha is {test_order.virtual_alpha} and virtual beta is {test_order.virtual_beta}')
    alpha_list, beta_list, a_list, virtual_alpha_list, virtual_beta_list, liquidity_list = [], [], [], [], [], []
    for a in [Dec(0.001), Dec(0.01), Dec(0.1), Dec(0.2), Dec(0.5)]:
        test_order = RangeOrder(Dec(1), NP.exp(-a), NP.exp(a))
        test_order.fix_alpha(Dec(1000.))
        a_list.append(a)
        alpha_list.append(test_order.reserve.alpha)
        beta_list.append(test_order.reserve.beta)
        virtual_alpha_list.append(test_order.virtual_alpha)
        virtual_beta_list.append(test_order.virtual_beta)
        liquidity_list.append(test_order.liquidity)

    import pandas as PD
    leverage_overview = PD.DataFrame({
        'a': a_list,
        'alpha': alpha_list,
        'beta': beta_list,
        'virtual_alpha': virtual_alpha_list,
        'virtual_beta': virtual_beta_list,
        'liquidity': liquidity_list
    })
    leverage_overview.to_csv('leverage_overview.csv')
    test_swap = Swap(beta=Dec(900))
    test_order.meet_swap(test_swap)
    print('After the first swap', test_order.reserve)
    test_order.meet_swap(test_swap)
    print('After the first swap', test_order.reserve)
    print('Finally, transaction fee is', test_order.transaction_fee)
