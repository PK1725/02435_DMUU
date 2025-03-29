import sys
sys.path.append("v2_Assignment_Codes")  # Add the folder to the search path

#load data
from v2_data import get_fixed_data
from PriceProcess import price_model
from WindProcess import wind_model

import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import *

data = get_fixed_data()

def generate_time_series(T):
    price = np.zeros(T)
    wind = np.zeros(T)
    wind[0] = data['wind_power']
    price[0] = data['price']
    for t in range(T-1):
        if t == 0:
            wind[t+1] = wind_model(data['wind_power'], data['wind_power_previous'], data)
            price[t+1] = price_model(data['price'], data['price_previous'],wind[t+1], data)   
        else:
            wind[t+1] = wind_model(wind[t], wind[t-1], data)
            price[t+1] = price_model(price[t], price[t-1],wind[t+1], data)
    return price, wind


def generate_experiment_series():
    # generate n truth scenarios to be used in all the tests
    T = data['num_timeslots']
    np.random.seed(0)
    n = 1000
    winds = np.zeros((n,T))
    prices = np.zeros((n,T))
    for i in range(n):
        prices[i], winds[i] = generate_time_series(T)
    return prices, winds