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


def solve_oih_model(price,wind,T,h_0,e_on_init,t_start=0): 

    #ranges
    T_range = range(T)
    Tm1_range = range(-1,T-1)

    # Create a model
    model = ConcreteModel()
    # Declare indexed variable for the price
    model.p_grid = Var(T_range, within=NonNegativeReals,name='p_grid')
    model.e_h2p = Var(T_range, within=NonNegativeReals,name='e_h2p')
    model.e_p2h = Var(T_range, within=NonNegativeReals,name='e_p2h')
    model.e_on = Var(Tm1_range, within=Binary,name='e_on')

    model.h = Var(range(T), within=NonNegativeReals,bounds=(0,data['hydrogen_capacity']),name='h')

    # Objective function
    def objective_rule(model):
        return sum(price[t] * model.p_grid[t] + data['electrolyzer_cost']*model.e_on[t-1] for t in range(T))

    model.profit = Objective(rule=objective_rule, sense=minimize)

    model.DemandConstraint = Constraint(T_range, rule=lambda model, t: model.p_grid[t] + wind[t] + data['conversion_h2p']*model.e_h2p[t] - model.e_p2h[t] >= data['demand_schedule'][t_start+t])

    # contraints

    model.h_contraint = Constraint(range(T-1),expr=lambda model, t: model.h[t+1] == model.h[t]+data['conversion_p2h']*model.e_p2h[t]-model.e_h2p[t])

    model.p2h_constraint = Constraint(T_range,rule=lambda model,t: model.e_h2p[t] <= model.h[t])
    model.p2h_constraint2 = Constraint(T_range,rule=lambda model,t: data['conversion_h2p']*model.e_h2p[t] <= data['h2p_max_rate'])

    model.conversion_contraint = Constraint(T_range,rule=lambda model,t: data['conversion_p2h'] * model.e_p2h[t] <= data['p2h_max_rate']*model.e_on[t-1])

    model.tank_start = Constraint(expr=model.h[0] == h_0)

    model.electrolyser_start = Constraint(expr=model.e_on[-1] == e_on_init)

    # Create a solver
    solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured

    # Solve the model
    results = solver.solve(model, tee=False)
    return results,model



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

def sim_optimal_in_hindsight(E,winds,prices):
    total_cost = 0
    values = []
    for i in tqdm(range(E), desc="Simulating OIH"):
        wind = winds[i]
        price = prices[i]
        _,model = solve_oih_model(price,wind,T,0,0)
        total_cost += value(model.profit)
        values.append(value(model.profit))
    return total_cost/E,values

from sklearn.cluster import KMeans

class realization():
    def __init__(self,t,local_scenario,wind,price,prev):
        self.t = t
        self.wind = wind
        self.price = price
        self.prev = prev
        self.local_scenario = local_scenario


def generate_scenarios(wind, price, wind_previous, price_previous, stages, k=4, n_samples=100):
    scenarios = []
    scenarios.append([realization(0, 0, wind, price, None)])
    scenario_probs = [[1.0]]

    for t in range(stages):
        new_scenarios = []
        new_probs = []
        for i, realization_i in enumerate(scenarios[t]):
            wind_samples = []
            price_samples = []
            for _ in range(n_samples):
                if t == 0:
                    wind_future = wind_model(realization_i.wind, wind_previous, data)
                    price_future = price_model(realization_i.price, price_previous, wind_future, data)
                else:
                    wind_future = wind_model(realization_i.wind, realization_i.prev.wind, data)
                    price_future = price_model(realization_i.price, realization_i.prev.price, wind_future, data)
                wind_samples.append(wind_future)
                price_samples.append(price_future)

            if k < n_samples:
                samples = np.column_stack((wind_samples, price_samples))
                kmeans = KMeans(n_clusters=k, random_state=0).fit(samples)
                centroids = kmeans.cluster_centers_
                labels = kmeans.labels_

                for j in range(k):
                    centroid_wind, centroid_price = centroids[j]
                    prob = np.mean(labels == j)
                    new_scenarios.append(realization(t+1, j, centroid_wind, centroid_price, realization_i))
                    new_probs.append(scenario_probs[t][i] * prob)
            else:
                for j in range(n_samples):
                    new_scenarios.append(realization(t+1, j, wind_samples[j], price_samples[j], realization_i))
                    new_probs.append(scenario_probs[t][i] / n_samples)

        scenarios.append(new_scenarios)
        scenario_probs.append(new_probs)

    return scenarios, scenario_probs

def scenarios_to_matrices(scenarios):
    T = len(scenarios)
    S = len(scenarios[-1])
    
    wind_matrix = np.zeros((T, S))
    price_matrix = np.zeros((T, S))
    a_set = [[]]*(T)
    a_set[0] = a_set[0] + [[i for i in range(S)]]
    for t in range(1,T):
        for s in range(len(scenarios[t])):
            span = int(len(scenarios[-1]) / len(scenarios[t]))
            prev_scenario = scenarios[t][s].prev
            a_set_indices = [_ for _ in range(s*span,(s+1)*span)]
            wind_matrix[t-1,span*s:(s+1)*span] = prev_scenario.wind
            price_matrix[t-1,span*s:(s+1)*span] = prev_scenario.price
            a_set[t] = a_set[t] + [a_set_indices]

    wind_matrix[-1] = [r.wind for r in scenarios[-1]]
    price_matrix[-1] = [r.price for r in scenarios[-1]]

    return wind_matrix, price_matrix,a_set


def solve_multi_stage_model(t_start,scenarios,a_set,scenario_probs,price,wind,h_0,e_on_0):
    T = len(scenarios)
    S = len(scenarios[-1])
    
    # Create a model
    model = ConcreteModel()
    # Declare indexed variable for the price
    model.p_grid = Var(range(T),range(S), within=NonNegativeReals,name='p_grid')
    model.e_h2p = Var(range(T),range(S), within=NonNegativeReals,bounds=(0,data['h2p_max_rate']),name='e_h2p')
    model.e_p2h = Var(range(T),range(S), within=NonNegativeReals,bounds=(0,data['p2h_max_rate']),name='e_p2h')
    model.e_on = Var(range(-1,T-1),range(S), within=Binary,name='e_on',initialize=0)

    model.h = Var(range(T),range(S), within=NonNegativeReals,bounds=(0,data['hydrogen_capacity']),name='h')

    vars = [model.p_grid,model.e_h2p,model.e_p2h,model.e_on,model.h]

    model.non_anticipativity = ConstraintList()
    for var in vars:
        for t in range(T-1):
            for s in range(S):
                # find s index in a_set[t]
                s_relative = s % len(a_set[t])
                for s_prime in a_set[t][s_relative]: 
                    model.non_anticipativity.add(var[t, s] == var[t, s_prime])

    # Objective function
    def objective_rule(model):
        return sum(scenario_probs[s]*(price[t,s] * model.p_grid[t,s] + data['electrolyzer_cost']*model.e_on[t-1,s]) for s in range(S) for t in range(T))

    model.profit = Objective(rule=objective_rule, sense=minimize)

    model.DemandConstraint = Constraint(range(T),range(S), rule=lambda model, t,s: model.p_grid[t,s] + wind[t,s] + data['conversion_h2p']*model.e_h2p[t,s] - model.e_p2h[t,s] >= data['demand_schedule'][t_start+t])

    # contraints
    model.h_contraint = Constraint(range(T-1), range(S), rule=lambda model, t, s: model.h[t+1, s] == model.h[t, s] + data['conversion_p2h']*model.e_p2h[t, s] - model.e_h2p[t, s])

    model.p2h_constraint = Constraint(range(T), range(S), rule=lambda model, t, s:  model.e_h2p[t, s] <= model.h[t, s])
    model.p2h_constraint2 = Constraint(range(T),range(S),rule=lambda model,t,s: data['conversion_h2p']*model.e_h2p[t, s] <= data['h2p_max_rate'])
    
    model.conversion_contraint = Constraint(range(T), range(S), rule=lambda model, t, s:  data['conversion_p2h']*model.e_p2h[t, s] <= data['p2h_max_rate']*model.e_on[t-1, s])

    model.tank_start = Constraint(range(S), rule=lambda model, s: model.h[0, s] == h_0)

    model.electrolyser_start = Constraint(range(S), rule=lambda model, s: model.e_on[-1, s] == e_on_0)

    # Create a solver
    solver = SolverFactory('gurobi') 

    # Solve the model
    results = solver.solve(model, tee=False)
    #print(f"S= {S}, T={T}, n_variables={len(list(model.component_objects(Var)))}")
    return results,model


class ValueFunction():
    def __init__(self, T, state_dim):
        self.T = T
        self.state_dim = state_dim
        self.weights = np.ones((T, state_dim + 1))  # +1 for the bias term
        
    def compute_value_explicit(self, t, state):
        # Append 1 for the bias term to the state
        if t >= T:
            return 0
        state_with_bias = state + [1]
        value = 0
        for j in range(len(state_with_bias)):
            value += state_with_bias[j] * self.weights[t, j].item()
        return value
    
    def compute_value(self, t, states):
        if t >= T:
            return np.zeros(states.shape[0])
        # Append 1 for the bias term to each state
        states_with_bias = np.hstack((states, np.ones((states.shape[0], 1))))
        return np.dot(states_with_bias, self.weights[t])

    def update(self, t, states, target_values):
        if t >= T:
            return 
        # Append 1 for the bias term to each state
        states_with_bias = np.hstack((states, np.ones((states.shape[0], 1))))
        # Solve the least squares problem to find the optimal weights
        self.weights[t], _, _, _ = np.linalg.lstsq(states_with_bias, target_values, rcond=None)
    
    def squared_error(self, t, states, target_values):
        # Compute the squared error
        predicted_values = self.compute_value(t, states)
        return np.mean((predicted_values - target_values) ** 2)
    
def sample_representative_state_pairs(I):
    T = data['num_timeslots']
    state_pairs = np.zeros((T,I,7)) # seven state variables
    for i in range(I):
        # sample exogenous state variables
        # TODO : should varying initial coniditions be used?
        price, wind = generate_time_series(T)
        # sample endogenous state variables
        h = np.random.uniform(0, data['hydrogen_capacity'], T)
        e_on = np.random.choice([0, 1], T)
        for t in range(T):
            state = [t, h[t], e_on[t-1] if t > 0 else 0, wind[t], wind[t-1] if t > 0 else data['wind_power_previous'], price[t], price[t-1] if t > 0 else data['price_previous']]
            state_pairs[t, i] = state
    return state_pairs
def value_minimization(V: ValueFunction,t,state_cur,scenarios, gamma,print_result=False): 

    t, h, e_on_tm1, wind, wind_previous, price, price_previous = state_cur

    # Create a model
    model = ConcreteModel()
    # Declare indexed variable for the price
    model.p_grid = Var(within=NonNegativeReals,name='p_grid')
    model.e_h2p = Var(within=NonNegativeReals,name='e_h2p')
    model.e_p2h = Var(within=NonNegativeReals,name='e_p2h')
    model.e_on = Var(within=Binary,name='e_on')
    
    # declare the new state
    model.next_e_on = Var(within=Binary,name='new_e_on')
    model.next_h = Var(within=NonNegativeReals,bounds=(0,data['hydrogen_capacity']),name='new_h')

    # Objective function
    def objective_rule(model):
        
        expected_next_value = 0
        for scenario in scenarios:
            scenario_state =  [t+1, model.next_h, model.next_e_on, scenario.wind, wind, scenario.price, price]
            expected_next_value += V.compute_value_explicit(int(t)+1,scenario_state)
        expected_next_value /= len(scenarios) 
        
        return price * model.p_grid + data['electrolyzer_cost']*model.e_on + gamma * expected_next_value

    model.profit = Objective(rule=objective_rule, sense=minimize)
    model.DemandConstraint = Constraint(rule=lambda model: model.p_grid + wind + data['conversion_h2p']*model.e_h2p - model.e_p2h >= data['demand_schedule'][int(t)])

    # contraints

    model.h_contraint = Constraint(expr=lambda model: model.next_h == h + data['conversion_p2h']*model.e_p2h-model.e_h2p)

    model.p2h_constraint = Constraint(rule=lambda model: model.e_h2p <= h)
    model.p2h_constraint2 = Constraint(rule=lambda model: data['conversion_h2p']*model.e_h2p <= data['h2p_max_rate'])

    model.conversion_contraint = Constraint(rule=lambda model: data['conversion_p2h'] * model.e_p2h <= data['p2h_max_rate']*e_on_tm1)

    model.e_on_constraint = Constraint(rule=lambda model: model.e_on == model.next_e_on)

    # Create a solver
    solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured

    # Solve the model
    results = solver.solve(model, tee=False)
    if print_result:
        # Check if an optimal solution was found
        if results.solver.termination_condition == TerminationCondition.optimal:
            print("Optimal solution found")
            print(f"profit: {value(model.profit)}")
            print(f"p_grid: {value(model.p_grid)}")
            print(f"e_h2p: {value(model.e_h2p)}")
            print(f"e_p2h: {value(model.e_p2h)}")
            print(f"e_on: {value(model.e_on)}")
        else:
            print("No optimal solution found.")
    decision = (model.e_on.value,model.e_p2h.value,model.e_h2p.value,model.p_grid.value)
    return decision,value(model.profit)

def backward_value_approx(V, state_pairs, K, data,gamma=0.9):
    T = data['num_timeslots']
    I = state_pairs.shape[1]
    for t in range(T-1, -1, -1):
        print(f"t={t}")
        value_targets = np.zeros(I)
        # go trough state pairs
        for i in range(I):
            state = state_pairs[t, i]
            _, h, e_on_tm1, wind, wind_previous, price, price_previous = state
            scenarios, scenario_probs = generate_scenarios(wind, price, wind_previous, price_previous, 1, k=K, n_samples=K)
            _,value_targets[i] = value_minimization(V, t, state, scenarios[0], gamma)
        print(V.squared_error(t, state_pairs[t], value_targets))
        V.update(t, state_pairs[t], value_targets)
        print(V.squared_error(t, state_pairs[t], value_targets))        
    return V



# ------ Policies ------

# policy: state -> decision
def dummy_policy(t,h,e_on,wind,wind_previous,price,price_previous,data):
    e_on = 0
    e_p2h = 0
    e_h2p = 0
    p_grid = max(data['demand_schedule'][t]-wind,0)
    return (e_on,e_p2h,e_h2p,p_grid)



class MultiStagePolicy:
    def __init__(self, L=4,k=3):
        self.L = L
        self.k = k

    def __call__(self, t, h, e_on, wind, wind_previous, price, price_previous,data):
        L = min(self.L, data['num_timeslots'] - t) - 1

        scenarios, scenario_probs = generate_scenarios(wind, price, wind_previous, price_previous, L, k=self.k)
        wind_matrix, price_matrix, a_set = scenarios_to_matrices(scenarios)
        results, model = solve_multi_stage_model(t, scenarios, a_set, scenario_probs[-1], price_matrix, wind_matrix, h, e_on)

        if not results.solver.termination_condition == TerminationCondition.optimal:
            raise ValueError(f"Optimal solution not found t={t}")
        
        e_on = value(model.e_on[0, 0]) if t < data['num_timeslots'] - 1 else 0
        e_p2h = value(model.e_p2h[0, 0])
        e_h2p = value(model.e_h2p[0, 0])
        p_grid = value(model.p_grid[0, 0])
        return e_on, e_p2h, e_h2p, p_grid
    

class ADPPolicy():
    def __init__(self, V, data,gamma = 0.9):
        self.V = V
        self.data = data
        self.gamma = gamma

    def __call__(self, t, h, e_on, wind, wind_previous, price, price_previous,data):
        scenarios, scenario_probs = generate_scenarios(wind, price, wind_previous, price_previous, 1)
        decision, _ = value_minimization(self.V, t, [t,h,e_on,wind,wind_previous,price,price_previous], scenarios[0], self.gamma)
        return decision
