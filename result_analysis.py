import pandas as PD
import numpy as NP
import os

def simulation_analysis(base_folder, pool_name) -> PD.DataFrame:
    global basic_analysis
    b_value, mean_stopping_time, rebase_times, win_times, mean_pnl, apr = [], [], [], [], [], []
    try:
        all_epoc_records = [_ for _ in os.listdir(base_folder) if _[-8: ] == 'epoc.csv']
        try:
            cost = basic_analysis.loc[basic_analysis['Pool'] == pool_name + '.csv']['mean_cost'].to_list()[0]
        except:
            cost = 15

        for single_b_str in all_epoc_records:
            try:
                b_value.append(round(float(single_b_str[: -9].replace('_', '.')), 2))
            except:
                continue
            single_result_path = base_folder + '/' + single_b_str
            single_result = PD.read_csv(single_result_path)
            print(f'{str(single_b_str)} and records as {str(len(single_result))}')

            single_result['epoc_lasts'] = single_result['epoc_end_index'] - single_result['epoc_begin_index']
            epoc_lasts = single_result['epoc_lasts'].values
            actual_stop_timing = NP.mean(epoc_lasts)
            mean_stopping_time.append(actual_stop_timing)

            # calculate rebase times, win times, etc
            single_result['epoc_pnl'] = single_result['end_wealth'] - single_result['holding_wealth']
            epoc_pnl = single_result['epoc_pnl'].values
            # return_ratio = NP.sum(epoc_pnl) / NP.sum(single_result['invest_wealth'].values)
            rebase_times.append(len(single_result))
            win_times.append(len(NP.where(epoc_pnl > 0)[0]))
            mean_pnl.append((NP.mean(epoc_pnl)))

            single_apr = (NP.sum(single_result['epoc_pnl']) - len(single_result) * cost) / NP.max(single_result['invest_wealth'])
            apr.append(single_apr)
        result_analysis = PD.DataFrame({
            'b_value': b_value,
            'mean_stopping_time': [round(float(_), 2) for _ in mean_stopping_time],
            'rebase_times': rebase_times,
            'win_times': win_times,
            'pnl': [round(float(_), 2) for _ in mean_pnl],
            'apr': [round(float(_), 2) for _ in apr]
        })
        result_analysis = result_analysis.sort_values(by='b_value').reset_index(drop=True)
        return result_analysis
    except:
        print('Wrong')
        raise Exception

if __name__ == '__main__':
    basic_analysis = PD.read_csv('para_result.csv')
    a = simulation_analysis('numerical_detail', 'simulation')
# source_path = 'Detailed/'
# output_path = 'Summary/'
# all_pools = [_ for _ in os.listdir(source_path) if _[-1] == '5']
# for single_pool_name in all_pools:
#
#         result_analysis.to_csv(f'{output_path}{single_pool_name}_summary.csv')
#     except:
#         print(single_pool_name)