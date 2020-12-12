import numpy as np
from sla import SLA
from app import App
import random
from itertools import product
from ortools.linear_solver import pywraplp


class Market:
    def __init__(self, apps):
        self.apps = apps

        self.period_length =  10 # length of period (minutes)

        ### CORE PARAMETERS
        # This market contains only Xeon cores with 2 power modes,
        # using data from Guevara.
        # mode 0 = sleep, mode 1 = active
        self.G = 1 # number machine groups
        self.M = 2 # number power modes
        self.MACHINES = [15] # number total machines in each group
        self.CORES = [4] # number cores per machine in each group
        self.cycles_per_transaction = 300 # MCycles per transaction
        
        # cycles supplied by cores (MCycles / minute)
        self.gamma = lambda g, t : [
            [0, 2.5]
        ][g][t] * 1000 * 60
        # time to switch power modes (minutes)
        self.delta = lambda g, f, t : [
            [0, 8],
            [6, 0]
        ][f][t] / 60

        self.E_mult = 1.6
        self.E_cost = .07 / (1000 * 60) # cost per unit energy ($ / (W * min))
        self.E_base_active = lambda g, tau, t : [25, 65][t] * tau # energy used for platform (W * min)
        self.E_base_idle = lambda g, tau, t : [25, 65][t] * tau # energy used for unsold platforms (W * min)
        self.E_trans = lambda g, f, t : [ # energy used for transition for each machine (W * min)
            [0, 65],
            [65, 0]
        ][f][t] * self.delta(g, f, t)
        self.E_core_active = lambda g, tau, t : [ # marginal energy for core (W * min)
            [0, 15.6]
        ][g][t] * tau
        self.E_core_idle = lambda g, tau, t : [ # marginal energy for unsold cores (W * min)
            [0, 7.8]
        ][g][t] * tau
        self.H_trans = lambda g, f, t : 0 if f == t else .05 # cost of transitioning ($)
        self.H_base = lambda g, tau, t : 0 if t == 0 else .0001 * tau # cost of maintainting machine ($), chosen arbitrarily

        self.current_time = 0 # start time of the next allocation period

        # allocation from the last completed period
        self.current_core_allocation = { app_id : { (g,f) : 0 for g, f in product(range(self.G), range(self.M)) } for app_id in self.apps }
        self.current_unsold_cores = { (g,f) : (0 if f != 0 else self.CORES[g] * self.MACHINES[g]) for g, f in product(range(self.G), range(self.M)) }

        # temp variables for storage
        self.saved_core_allocation = None
        self.saved_unsold_cores = None


    '''
    Runs the allocation for the period starting at current_time.
    Effects:
        - Updates current_core_allocation, current_unsold_cores, current_time, and each app's current price/cycles
        - Returns welfare of this period's allocation, user values, VCG prices
    Inputs:
        - use_VCG_price : bool or list; Indicates which apps should pay the VCG price (True = all, False = none)
        - allow_holding : bool; Indicates whether apps can choose to hold their cores from last period (done greedily by the app)
    '''
    def get_allocation_and_prices(self, use_VCG_price=False, allow_holding=False):
        # see if each user will hold their last period allocation
        holding=[]
        for app_id, app in self.apps.items():
            if allow_holding and app.will_hold(self.current_time, self.period_length):
                holding.append(app_id)

        # realize the demands for the current time
        for app in self.apps.values():
            app.set_demand(self.current_time)

        # run the actual allocation and calculate VCG pricing
        VCG = {}
        total_welfare, total_values, total_cycles = self.allocate_all_apps(holding)
        for app_id in self.apps:
            welfare, _, _ = self.allocate_without_app(app_id, holding)
            price = welfare - (total_welfare - total_values[app_id])
            VCG[app_id] = price

        # store price and allocated cycles in each app
        for app_id in self.apps:
            if app_id in holding: # don't change price or alloc if holding
                continue
            if (type(use_VCG_price) == bool and use_VCG_price) or (type(use_VCG_price) != bool and app_id in use_VCG_price):
                self.apps[app_id].current_price = VCG[app_id]
            else:
                self.apps[app_id].current_price = total_values[app_id]
            self.apps[app_id].current_cycles = total_cycles[app_id]

        # finalize current allocation in the datacenter
        self.current_core_allocation = self.saved_core_allocation
        self.current_unsold_cores = self.saved_unsold_cores
        self.saved_core_allocation = None
        self.saved_unsold_cores = None

        # advance the time to next period
        self.current_time += self.period_length
        
        return total_welfare, total_values, VCG

    def allocate_all_apps(self, holding=[]):
        return self.market_allocate(True, [], holding)

    def allocate_without_app(self, id_to_exclude, holding=[]):
        return self.market_allocate(False, [id_to_exclude], holding)

    def market_allocate(self, save_alloc=False, exclude_from_obj=[], holding=[]):

        solver = pywraplp.Solver.CreateSolver('SCIP')
        infinity = solver.infinity()

        C_current = {}
        for g, f in product(range(self.G), range(self.M)):
            C_current[(g, f)] = sum([self.current_core_allocation[app_id][(g, f)] for app_id in self.apps]) + self.current_unsold_cores[(g, f)]

    
        #############################################
        # Define MIP
        #############################################
    
        # buyer valuation model
        V = {} # value provided to application a V = F(Q) ($)
        Q = {} # cycles provided to application a (MCycles)
        Q_seq = {} # sequential cycles provided to application a
        Q_par = {} # parallel cycles provided to application a
        Z_seg = {} # segment indicator for piecewise linear function
        Q_seg = {} # segment value for piecewise linear function
        C_sold = {}
        C_sold_seq = {} # sold cores used sequentially
        C_unsold = {}
        C_partunsold = {}
        for a, A in self.apps.items():
            V[a] = solver.NumVar(0, infinity, f'V[{a}]')
            Q[a] = solver.NumVar(0, infinity, f'Q[{a}]')
            Q_seq[a] = solver.NumVar(0, infinity, f'Q_seq[{a}]')
            Q_par[a] = solver.NumVar(0, infinity, f'Q_par[{a}]')
    
            # set measured application demand for A
            A.sla.compute_supply_value(A.current_demand, self.period_length)
    
            # auxiliary for computing V = F(Q)
            Z_seg[a] = {}
            Q_seg[a] = {}
            x_prev = lambda s : 0 if s == 0 else A.sla.supply_x[s-1] * self.cycles_per_transaction * self.period_length
            x_curr = lambda s : A.sla.supply_x[s] * self.cycles_per_transaction * self.period_length
            y_prev = lambda s : 0 if s == 0 else A.sla.value_y[s-1]
            y_curr = lambda s : A.sla.value_y[s]
            # iterate over segments
            num_seg = len(A.sla.supply_x)
            for s in range(num_seg):
                Z_seg[a][s] = solver.IntVar(0, 1, f'Z_seg[{a}][{s}]')
                Q_seg[a][s] = solver.NumVar(0, infinity, f'Q_seg[{a}][{s}]')
                # ^Q_{s-1} Z_s <= Q_s <= ^Q_{s} Z_s
                solver.Add(x_prev(s) * Z_seg[a][s] <= Q_seg[a][s])
                solver.Add(x_curr(s) * Z_seg[a][s] >= Q_seg[a][s])
            # Z_seg[a][s] sum to 1
            solver.Add(sum(Z_seg[a][s] for s in range(num_seg)) == 1)
            # Q_seg[a][s] sum to Q[a]
            solver.Add(sum(Q_seg[a][s] for s in range(num_seg)) == Q[a])
            # set V = F(Q) for piecewise linear F
            solver.Add(sum(
                y_prev(s) * Z_seg[a][s] + 
                (y_curr(s) - y_prev(s)) / (x_curr(s) - x_prev(s)) * 
                (Q_seg[a][s] - x_prev(s) * Z_seg[a][s])
                for s in range(num_seg)
            ) == V[a])
    
            # set Q
            C_sold[a] = {}
            C_sold_seq[a] = {}
            for g, f, t in product(range(self.G), range(self.M), range(self.M)):
                C_sold[a][(g,f,t)] = solver.IntVar(0, infinity, f'C_sold[{a}][{(g,f,t)}]')
                C_sold_seq[a][(g,f,t)] = solver.IntVar(0, 1, f'C_sold_seq[{a}][{(g,f,t)}]')

                # need to have bought sequential core
                solver.Add(C_sold_seq[a][(g,f,t)] <= C_sold[a][(g,f,t)])
            #solver.Add(C_sold[0][(0,0,1)] == 4)

            # each application can use at most one core sequentially
            solver.Add(sum(
                C_sold_seq[a][(g,f,t)]
                for g, f, t in product(range(self.G), range(self.M), range(self.M))
            ) <= 1)

            # set number of sequential cycles provided
            solver.Add(sum(
                self.gamma(g, t) * (self.period_length - self.delta(g, f, t)) * C_sold_seq[a][(g,f,t)]
                for g, f, t in product(range(self.G), range(self.M), range(self.M))
            ) == Q_seq[a])

            # set number of parallel cycles provided
            solver.Add(sum(
                self.gamma(g, t) * (self.period_length - self.delta(g, f, t)) * C_sold[a][(g,f,t)]
                for g, f, t in product(range(self.G), range(self.M), range(self.M))
            ) == Q_par[a])
            
            # set number of cycles provided
            solver.Add(
                (1 - A.parallel_fraction) * Q_seq[a] + A.parallel_fraction * Q_par[a]
            == Q[a])
    
        # goods in the market
        M_sold = {}
        M_unsold = {}
        for g, f, t in product(range(self.G), range(self.M), range(self.M)):
            M_sold[(g,f,t)] = solver.IntVar(0, infinity, f'M_sold[{(g,f,t)}]')
            M_unsold[(g,f,t)] = solver.IntVar(0, infinity, f'M_unsold[{(g,f,t)}]')
            C_unsold[(g,f,t)] = solver.IntVar(0, infinity, f'C_unsold[{(g,f,t)}]')
        for g, t in product(range(self.G), range(self.M)):
            C_partunsold[(g,t)] = solver.IntVar(0, infinity, f'C_partunsold[{(g,t)}]')
    
        for g, t in product(range(self.G), range(self.M)):
            solver.Add(
                sum(self.CORES[g] * M_sold[(g,f,t)] for f in range(self.M)) == 
                sum(
                    C_sold[a][(g,f,t)] 
                    for a, f in product(self.apps.keys(), range(self.M))
                ) + C_partunsold[(g,t)]
            )
            solver.Add(
                sum(self.CORES[g] * M_unsold[(g,f,t)] for f in range(self.M)) == 
                sum(C_unsold[(g,f,t)] for f in range(self.M)) - C_partunsold[(g,t)]
            )
        for g in range(self.G):
            solver.Add(
                self.MACHINES[g] == sum(
                    M_sold[(g,f,t)] + M_unsold[(g,f,t)]
                    for f, t in product(range(self.M), range(self.M))
                )
            )
        for g, f in product(range(self.G), range(self.M)): # maintain consistency with current core states
            solver.Add( sum( C_sold[a][(g,f,t)] for a, t in product(self.apps.keys(), range(self.M))) 
                    + sum( C_unsold[(g,f,t)] for t in range(self.M) )
                    == C_current[(g, f)] )

        # keep allocation of those holding
        for a in holding:
            for g, f, t in product(range(self.G), range(self.M), range(self.M)):
                if f == t:
                    solver.Add( C_sold[a][(g,f,f)] == self.current_core_allocation[a][(g,f)] )
                else:
                    solver.Add( C_sold[a][(g,f,t)] == 0 )

    
        # seller cost model
        E = solver.NumVar(0, infinity, 'E')
        H = solver.NumVar(0, infinity, 'H')
        solver.Add(
            E == sum(
                self.E_mult * (self.E_trans(g,f,t) + self.E_base_active(g, self.period_length, t)) * 
                M_sold[(g,f,t)]
                for g, f, t in product(range(self.G), range(self.M), range(self.M)) 
            ) + sum(
                self.E_mult * self.E_core_active(g, self.period_length, t) * C_sold[a][(g,f,t)]
                for a, g, f, t in product(self.apps.keys(), range(self.G), range(self.M), range(self.M)) 
            ) + sum(
                self.E_mult * (self.E_trans(g,f,t) + self.E_base_idle(g, self.period_length, t)) *
                M_unsold[(g,f,t)]
                for g, f, t in product(range(self.G), range(self.M), range(self.M))
            ) + sum(
                self.E_mult * self.E_core_idle(g, self.period_length, t) * C_unsold[(g,f,t)]
                for g,f,t in product(range(self.G), range(self.M), range(self.M))
            )
        )
        solver.Add(
            H == sum(
                (self.H_trans(g,f,t) + self.H_base(g, self.period_length, t)) * (M_sold[(g,f,t)] + M_unsold[(g,f,t)])
                for g, f, t in product(range(self.G), range(self.M), range(self.M))
            )
        )
    
        #print('Number of variables =', solver.NumVariables())
        #print('Number of constraints =', solver.NumConstraints())
    
        # objective ($)
        solver.Maximize(
            sum(V[a] for a in self.apps.keys() if a not in exclude_from_obj) - (self.E_cost * E) - H
        )
    
        status = solver.Solve()

        #print(self.apps[0].sla.eval_value(Q[0].solution_value() / (self.period_length * self.cycles_per_transaction), self.apps[0].current_demand))
   
        if status == pywraplp.Solver.OPTIMAL:
            #print('Solution:')
            #print('Objective value =', solver.Objective().Value())
            if save_alloc:
                self.saved_core_allocation = { a : {} for a in self.apps.keys() }
                self.saved_unsold_cores = {}
                for a, g, t in product(self.apps.keys(), range(self.G), range(self.M)):
                    self.saved_core_allocation[a][(g,t)] = sum([C_sold[a][(g,f,t)].solution_value() for f in range(self.M)])
                    self.saved_unsold_cores[(g,t)] = sum([C_unsold[(g,f,t)].solution_value() for f in range(self.M)])
            return solver.Objective().Value(), {a : V[a].solution_value() for a in V}, {a : [Q[a].solution_value(), Q_seq[a].solution_value(), Q_par[a].solution_value()] for a in Q}

        else:
            print('The problem does not have an optimal solution.')
