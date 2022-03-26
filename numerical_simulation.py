# %%
from decimal import Decimal as Dec
import os
import cal_key_paras as KP
import v3_core as VC
import numpy as NP
import pandas as PD
import matplotlib.pyplot as plt
plt.style.use('seaborn')
NP.random.seed(1234)


class PriceSeries:

    def __init__(self, change_mean, change_variance):
        self.change_mean = change_mean  # required to be zero under math
        # for change variance, the smaller, the better
        self.change_variance = change_variance

        self.initial_price = NP.exp(NP.random.normal(change_mean, change_variance))
        self.current_price = self.initial_price

    def generate_next_price(self) -> tuple:
        random_direction = NP.random.normal(
            self.change_mean, self.change_variance)
        self.current_price *= NP.exp(random_direction)
        return self.current_price, random_direction


class UsedPriceSeries:

    def __init__(self, price_series: NP.array):
        self.price_series = price_series
        self.current_location = 0
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

    # when price out of break out range,rebaase interval
    def signal(self, price) -> bool:
        if price > self.break_point_high or price < self.break_point_low:
            return False
        return True


# %%
if __name__ == '__main__':

    N_price = 2000

    test_var = 4e-3

    test_price_series = PriceSeries(0,test_var)

    PD.DataFrame({

    'price': [test_price_series.generate_next_price()[0] for _ in range(int(200000))]
     }).to_csv('simulation_price.csv')
    store_path = 'numerical_detail/'

    if not os.path.exists(store_path):
        os.mkdir(store_path)

    
    #test_price_series = PriceSeries(2000, test_var)
    #N_price = 1000
    # price_records = []
    # for i in range(N_price):
    #     price_records.append(float(test_price_series.generate_next_price()[0]))
    # # price_records = [float(test_price_series.generate_next_price()[0])
    # #                 for _ in range(N_price)]
    rebase_grid= NP.arange(0.2,1,0.5)

    # %%
    # a:range order size , b:breakout interval size ,assume symmetric
    # a, b = Dec(10*test_var), Dec(100*test_var)
    # try: 
    APR,reset_times, total_reset_cost,  total_swap_fees, optim_a,optim_b =[], [], [ ],[ ], [ ],[ ]
    total_swap_times = 100 #15000 times
    reserve_alpha_info=PD.DataFrame(NP.zeros(total_swap_times))
    total_pnl,total_cum_pnl, total_wealth_change=PD.DataFrame(NP.zeros(total_swap_times-1)),PD.DataFrame(NP.zeros(total_swap_times-2)), PD.DataFrame(NP.zeros(total_swap_times-1))


    a_grid = NP.arange(0.5*test_var, 1*test_var,0.5*test_var)
    b_grid = NP.arange(1*test_var,3*test_var,1*test_var)

    for rebase_cost in rebase_grid :
        for a in a_grid:

            print("processing gas cost :",rebase_cost)
            print("processing a :", a)
            
                
            for b in b_grid:
                print("processing b:", b)

                
            

                used_price_series = UsedPriceSeries(PD.read_csv('simulation_price.csv')['price'].values)
                #used_price_series = UsedPriceSeries(price_records)

                initial_price = used_price_series.current_price
                
                deposit_alpha = Dec(10.)
                test_user = {
                    'strategy': Strategy(initial_price * Dec(NP.exp(-b)), initial_price * Dec(NP.exp(b))),
                    'reserves': VC.TokenInfo(deposit_alpha, Dec(0)),
                    'range_order': VC.RangeOrder(initial_price, initial_price * Dec(NP.exp(-a)), initial_price * Dec(NP.exp(a)))
                }

                test_user['range_order'].fix_alpha(test_user['reserves'].alpha)

                # print(f'liquidity {range_order.liquidity} virtual alpha {range_order.virtual_alpha} virtual beta {range_order.virtual_beta}')
                range_order = test_user['range_order']
                initial_reserve = {
                    'alpha': range_order.reserve.alpha,
                    'beta': range_order.reserve.beta
                }


                reset_record, cum_pnl, pnl, wealth_change, price_records, swap_records, reserve_alpha, reserve_beta, fee_alpha, fee_beta, token_wealth, fee_wealth =[], [], [],[],  [initial_price], [''], [
                    initial_reserve['alpha']], [initial_reserve['beta']], [Dec(0.)], [Dec(0.)], [initial_reserve['alpha'] + initial_reserve['beta'] * initial_price], [Dec(0.)]
                epoc_swap_begin_index, epoc_invest_price, epoc_invest_price_high, epoc_invest_price_low, epoc_invest_alpha, epoc_invest_beta, epoc_swap_end_index, epoc_end_price, epoc_end_alpha_pool, epoc_end_beta_pool, epoc_end_alpha_fee, epoc_end_beta_fee, epoc_begin_wealth, epoc_end_wealth = [
                    Dec(0)], [initial_price], [test_user['range_order'].price_high], [test_user['range_order'].price_low], [initial_reserve['alpha']], [initial_reserve['beta']], [], [], [], [], [], [], [test_user['range_order'].wealth], []

                
                
                #initialize variable
                reset=0  #number of reset price interval
                total_gas = 0

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

                    
                    #price_records.append(used_price_series.current_price)
                    

                    test_user['range_order'].meet_swap(one_swap)

                    swap_records.append(swap_info)
                    reserve_alpha.append(test_user["range_order"].reserve.alpha)
                    reserve_beta.append(test_user["range_order"].reserve.beta)

                    fee_alpha.append(
                        test_user['range_order'].transaction_fee.alpha)
                    fee_beta.append(test_user['range_order'].transaction_fee.beta)
                    token_wealth.append(test_user['range_order'].wealth)

                
                    fee_wealth.append(test_user['range_order'].transaction_fee.alpha +
                                    used_price_series.current_price * test_user['range_order'].transaction_fee.beta)


                    
                    if not(test_user['strategy'].signal(used_price_series.current_price)):

                        #cost include: mint and burn,

                        total_gas+=rebase_cost*2
                        reset+=1
                        reset_info=f'reset when swap time:{swap_times}'
                        reset_record.append(reset_info)
                        
                        #swap_records.append('rebase')

                        
                        # print(
                        #     f'\nRebase! Swap time {swap_times}\n------------------\n')

                        # epoc_swap_end_index.append(swap_times)
                        # epoc_end_alpha_pool.append(
                        #     test_user["range_order"].reserve.alpha)
                        # epoc_end_alpha_fee.append(
                        #     test_user['range_order'].transaction_fee.alpha)
                        # epoc_end_beta_pool.append(
                        #     test_user["range_order"].reserve.beta)
                        # epoc_end_beta_fee.append(
                        #     test_user['range_order'].transaction_fee.beta)
                        # epoc_end_price.append(used_price_series.current_price)
                        

                        #reset range order and break-out interval
                        initial_price = used_price_series.current_price
                        test_user = {
                            'strategy': Strategy(initial_price * Dec(NP.exp(-b)), initial_price * Dec(NP.exp(b))),
                            'reserves': VC.TokenInfo(reserve_alpha[-1], Dec(0)),
                            'range_order': VC.RangeOrder(initial_price, initial_price * Dec(NP.exp(-a)), initial_price * Dec(NP.exp(a)))
                        }
                        #reserve_alpha[-1],指用之前池子里面剩下的alpha token

                        test_user['range_order'].fix_alpha(test_user['reserves'].alpha)
                        range_order = test_user['range_order']
                        initial_reserve = {
                            'alpha': range_order.reserve.alpha,
                            'beta': range_order.reserve.beta
                        }

                    #finish of one swap
                    if not(swap_times ==0):
                    
                        pnl.append(token_wealth[-1]-token_wealth[-2] + fee_wealth[-1]- Dec(total_gas))
                        wealth_change=NP.diff(token_wealth)

                    if len(pnl)>1:
                        #ensure pnl has at least 2 number
                    
                        cum_pnl.append(pnl[-1]+pnl[-2])


                #finish of swap 
                reset_times.append(reset)
                pnl=PD.DataFrame(pnl)
                total_pnl=PD.concat([total_pnl,pnl],axis=1)
                
                cum_pnl=PD.DataFrame(cum_pnl)
                total_cum_pnl=PD.concat([total_cum_pnl,cum_pnl],axis=1)
                total_reset_cost.append(total_gas)
                total_swap_fees.append(sum(fee_wealth))
                wealth_change=PD.DataFrame(wealth_change)
                total_wealth_change=PD.concat([total_wealth_change,wealth_change],axis=1)

                reserve_alpha=PD.DataFrame(reserve_alpha)
                reserve_alpha_info=PD.concat([reserve_alpha_info,reserve_alpha],axis=1)
                #APR.append(pnl.iloc[:,0]/token_wealth[0:len(token_wealth)-2])

                
                

                
                
        #Finish one rebase_cost search
        #optim_a.append(a_grid[(NP.argmax(pnl)-1)//len(a_grid) -1 ])
        #optim_b.append( b_grid[(NP.argmax(pnl)-1)%len(b_grid) -1])
    
    #Finish of all rebase_cost search
    
 # %%   
    #print(final_profit)
    plt.plot(pnl[60:80])
    plt.xlabel('break-out interval index', fontsize=12)
    plt.ylabel('profit', fontsize=12)
    plt.title("gas price=1")
    plt.show()



                    
            #         epoc_swap_begin_index.append(swap_times + 1)
            #         epoc_invest_alpha.append(
            #             test_user["range_order"].reserve.alpha)
            #         epoc_invest_beta.append(test_user['range_order'].reserve.beta)
            #         epoc_invest_price.append(used_price_series.current_price)
            #         epoc_invest_price_high.append(
            #             test_user['range_order'].price_high)
            #         epoc_invest_price_low.append(
            #             test_user['range_order'].price_low)

            #         reserve_alpha.append(test_user["range_order"].reserve.alpha)
            #         reserve_beta.append(test_user["range_order"].reserve.beta)
            #         fee_alpha.append(
            #             test_user['range_order'].transaction_fee.alpha)
            #         fee_beta.append(test_user['range_order'].transaction_fee.beta)
            #         token_wealth.append(test_user['range_order'].wealth)
            #         fee_wealth.append(test_user['range_order'].transaction_fee.alpha +
            #                         used_price_series.current_price * test_user['range_order'].transaction_fee.beta)

            # epoc_swap_end_index.append(total_swap_times + 1)
            # epoc_end_alpha_pool.append(test_user["range_order"].reserve.alpha)
            # epoc_end_alpha_fee.append(
            #     test_user['range_order'].transaction_fee.alpha)
            # epoc_end_beta_pool.append(test_user["range_order"].reserve.beta)
            # epoc_end_beta_fee.append(test_user['range_order'].transaction_fee.beta)
            # epoc_end_price.append(used_price_series.current_price)

            # transaction_records = PD.DataFrame({
            #     'swap_index': [_ for _ in range(len(price_records))],
            #     'current_price': price_records,
            #     'single_swap': swap_records,
            #     'reserve_alpha': reserve_alpha,
            #     'reserve_beta': reserve_beta,
            #     'fee_alpha': fee_alpha,
            #     'fee_beta': fee_beta,
            #     'token_wealth': token_wealth,
            #     'fee_wealth': fee_wealth
            # })
            # transaction_records['wealth_hold'] = initial_reserve['alpha'] + \
            #     initial_reserve['beta'] * \
            #     transaction_records['current_price'].values
            # transaction_records['wealth_pool'] = transaction_records['token_wealth'] + \
            #     transaction_records['fee_wealth']

            # epoc_records = PD.DataFrame({
            #     'epoc_begin_index': epoc_swap_begin_index,
            #     'epoc_end_index': epoc_swap_end_index,
            #     'epoc_invest_alpha': epoc_invest_alpha,
            #     'epoc_invest_beta': epoc_invest_beta,
            #     'epoc_invest_price_high': epoc_invest_price_high,
            #     'epoc_invest_price_low': epoc_invest_price_low,
            #     'epoc_invest_price': epoc_invest_price,
            #     'epoc_end_alpha_pool': epoc_end_alpha_pool,
            #     'epoc_end_beta_pool': epoc_end_beta_pool,
            #     'epoc_end_alpha_fee': epoc_end_alpha_fee,
            #     'epoc_end_beta_fee': epoc_end_beta_fee,
            #     'epoc_end_price': epoc_end_price
            # })
            # epoc_records['invest_wealth'] = epoc_records['epoc_invest_alpha'] + \
            #     epoc_records['epoc_invest_beta'] * \
            #     epoc_records['epoc_invest_price']
            # epoc_records['end_wealth_in_pool'] = epoc_records['epoc_end_alpha_pool'] + \
            #     epoc_records['epoc_end_beta_pool'] * epoc_records['epoc_end_price']
            # epoc_records['end_wealth_fee'] = epoc_records['epoc_end_alpha_fee'] + \
            #     epoc_records['epoc_end_beta_fee'] * epoc_records['epoc_end_price']
            # epoc_records['end_wealth'] = epoc_records['end_wealth_in_pool'] + \
            #     epoc_records['end_wealth_fee']
            # epoc_records['holding_wealth'] = epoc_records['epoc_invest_alpha'] + \
            #     epoc_records['epoc_invest_beta'] * epoc_records['epoc_end_price']

            # # break
            # transaction_records.to_csv(
            #     f'{store_path}{str(round(float(b),2)).replace(".", "_")}_transaction.csv')
            # epoc_records.to_csv(
            #     f'{store_path}{str(round(float(b),2)).replace(".", "_")}_epoc.csv')
            # '''




# %%
