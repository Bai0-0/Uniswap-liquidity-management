import pandas as PD
from numerical_simulation import PriceSeries

test_price_series = PriceSeries(0, 0.05**2)
PD.DataFrame({
    'price': [test_price_series.generate_next_price()[0] for _ in range(int(20000000))]
}).to_csv('simulation_price.csv')