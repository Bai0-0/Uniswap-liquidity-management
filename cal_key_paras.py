Integral = .5
price_records = None

import numpy as NP


# import numerical_simulation as NS
# # test plot
# price_series = []
# price_generator = NS.PriceSeries(0., 0.005)
# while len(price_series) < 100000:
#     price_series.append(price_generator.generate_next_price()[0])
# price_changes = NP.diff(NP.log(NP.array(price_series)))
# import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
# plt.hist(price_changes, bins=10, normed=0)

# price_series = price_records

def cal_moment(price_series):
    # price_series = price_records

    price_changes = NP.diff(NP.log(NP.array(price_series)))
    return NP.mean(NP.abs(price_changes)), NP.var(price_changes)

def cal_psi(price_series, rate=0.003):
    moments = cal_moment(price_series)
    return rate * 2 * moments[0] / moments[1] * Integral

# cal_psi(price_series, 0.003)