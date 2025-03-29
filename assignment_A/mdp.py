import sys
sys.path.append("v2_Assignment_Codes")  # Add the folder to the search path

#load data
from v2_data import get_fixed_data
from PriceProcess import price_model
from WindProcess import wind_model
from utils import generate_time_series,generate_experiment_series

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pyomo.environ import *

data = get_fixed_data()
T = data['num_timeslots']

def check_feasibility(state,decison,data):
    t,h,e_on_tm1,wind,wind_previous,price,price_previous = state
    e_on_t,e_p2h,e_h2p,p_grid = decison
    # check demand constraint
    if not (data['demand_schedule'][t] <= 
            p_grid+wind+data['conversion_h2p']*e_h2p-data['conversion_p2h']*e_p2h
            or np.isclose(data['demand_schedule'][t], p_grid+wind+data['conversion_h2p']*e_h2p-data['conversion_p2h']*e_p2h)):
        raise ValueError(f"Demand constraint violated, t={t}, demand={data['demand_schedule'][t]} was {p_grid+wind+data['conversion_h2p']*e_h2p-data['conversion_p2h']*e_p2h}")
    # check tank constraint
    if not (e_h2p <= data['hydrogen_capacity']):
        raise ValueError(f"Tank constraint violated, t={t}, cap={data['hydrogen_capacity']} was {e_h2p}")
    # check p2h conversion constraint
    if not (data['conversion_p2h']*e_p2h <= data['p2h_max_rate']*e_on_tm1):
        raise ValueError(f"p2h conversion constraint violated, t={t}, cap={data['p2h_max_rate']*e_on_tm1} was {e_p2h}")
    # check h2p conversion constraint
    if not (data['conversion_h2p']*e_h2p <= data['h2p_max_rate']):
        raise ValueError(f"h2p conversion constraint violated, t={t}, cap={data['h2p_max_rate']} was {e_h2p}")
    # check h2p supply constraint
    if not (e_h2p <= h):
        raise ValueError(f"h2p supply constraint violated, t={t}, cap={h} was {e_h2p}")
    # check domains
    if not (e_on_t in [0,1]):
        raise ValueError('e_activate not in [0,1]')
    if not (e_p2h >= 0 and e_h2p >= 0 and p_grid >= 0):
        raise ValueError('e_p2h,e_h2p,p_grid not >= 0')
    

def sim_MDP_exp(policy,wind,price):
    T = data['num_timeslots']
    cost = 0

    # state variables
    h = np.zeros(T)
    
    # decision variables
    e_on = np.zeros(T)
    e_p2h = np.zeros(T)
    e_h2p = np.zeros(T)
    p_grid = np.zeros(T)

    for t in range(T):

        ### get state
        state = (t,h[t],e_on[t-1],wind[t],wind[t-1],price[t],price[t-1]) if t > 0 else (t,h[t],0,wind[t],data['wind_power_previous'],price[t],data['price_previous'])
        
        # decision update
        decision = policy(*state,data)
        e_on[t],e_p2h[t],e_h2p[t],p_grid[t] = decision
        # check feasibility of decision
        check_feasibility(state,decision,data)

        ### state update based on decision

        # update hydrogen tank
        if t+1 < T: h[t+1] = h[t] + data['conversion_p2h']*e_p2h[t] - e_h2p[t]

        ### update cost
        cost += price[t]*p_grid[t] + data['electrolyzer_cost']*e_on[t-1]

    return cost,e_on,e_p2h,e_h2p,p_grid,h,price,wind

def sim_MDP(E, policy, winds, prices):
    total_cost = 0
    costs = []
    for i in tqdm(range(E), desc="Simulating MDP"):
        wind = winds[i]
        price = prices[i]
        cost, e_on, e_p2h, e_h2p, p_grid, h, price, wind = sim_MDP_exp(policy, wind, price)
        costs.append(cost)
        total_cost += cost
    return total_cost / E,costs