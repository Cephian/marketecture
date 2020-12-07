import numpy as np
from sla import SLA
from app import App
import random
from itertools import product
from ortools.linear_solver import pywraplp

class Market:
    # TODO use a dictionary for params instead?
    def __init__(self, apps, G, M, CORES, MACHINES, gamma, delta, period_length=10):
        self.G = G # number machine groups
        self.M = M # number power modes, TODO change to be diff per group
        self.CORES = CORES # number cores per machine in each group, TODO change to be diff per group
        self.MACHINES = MACHINES # number total machines in each group, TODO change to be diff per group
        self.gamma = gamma
        self.delta = delta
        self.period_length = period_length

        self.apps = apps
        self.current_time = 0
        self.set_demands()

        self.current_core_allocation = { app_id : { (g,f) : 0 for g, f in product(range(self.G), range(self.M)) } for app_id in self.apps }
        self.current_unsold_cores = { (g,f) : (0 if f != 0 else self.CORES * self.MACHINES) for g, f in product(range(self.G), range(self.M)) }

        self.saved_core_allocation = {}
        self.saved_unsold_cores = {}

    def advance_time(self):
        self.current_time += self.period_length
        self.set_demands()

        self.current_core_allocation = self.saved_core_allocation
        self.current_unsold_cores = self.saved_unsold_cores
        self.saved_core_allocation = {}
        self.saved_unsold_cores = {}

    def set_demands(self):
        for app in self.apps.values():
            app.set_demand(self.current_time)

    def get_VCG_allocation_and_prices(self):
        VCG = {}
        total_welfare, total_values = self.allocate_all_apps()
        for app_id in self.apps:
            welfare, values = self.allocate_without_app(app_id)
            price = welfare - (total_welfare - total_values[app_id])
            VCG[app_id] = price
        return total_welfare, total_values, VCG

    def allocate_all_apps(self):
        return self.market_allocate(self.apps, True)

    def allocate_without_app(self, id_to_exclude):
        return self.market_allocate({app_id : app for app_id, app in self.apps.items() if app_id != id_to_exclude})

    def market_allocate(self, applications, save_alloc=False):
        solver = pywraplp.Solver.CreateSolver('SCIP')
        infinity = solver.infinity()

        C_current = {}
        for g, f in product(range(self.G), range(self.M)):
            C_current[(g, f)] = sum([self.current_core_allocation[app_id][(g, f)] for app_id in self.apps]) + self.current_unsold_cores[(g, f)]
    
        #############################################
        # Define MIP
        #############################################
    
        # buyer valuation model
        V = {} # value provided to application a V = F(Q)
        Q = {} # cycles provided to application a
        Q_seq = {} # sequential cycles provided to application a
        Q_par = {} # parallel cycles provided to application a
        Z_seg = {} # segment indicator for piecewise linear function
        Q_seg = {} # segment value for piecewise linear function
        C_sold = {}
        C_sold_seq = {} # sold cores used sequentially
        C_sold_par = {} # sold cores used in parallel
        C_unsold = {}
        C_partunsold = {}
        for a, A in applications.items():
            V[a] = solver.NumVar(0, infinity, f'V[{a}]')
            Q[a] = solver.NumVar(0, infinity, f'Q[{a}]')
            Q_seq[a] = solver.NumVar(0, infinity, f'Q_seq[{a}]')
            Q_par[a] = solver.NumVar(0, infinity, f'Q_par[{a}]')
    
            # set measured application demand for A
            A.parallel_fraction = 1 / random.choice([1, 2, 10, 20, 30, 35, 70])
            assert (0 <= A.parallel_fraction) and (A.parallel_fraction <= 1), \
                'Parallel fraction of application must be in the interval [0, 1].'
            A.sla.compute_supply_value(A.current_demand, self.period_length)
    
            # auxiliary for computing V = F(Q)
            Z_seg[a] = {}
            Q_seg[a] = {}
            x_prev = lambda s : 0 if s == 0 else A.sla.supply_x[s-1]
            x_curr = lambda s : A.sla.supply_x[s]
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
            C_sold_par[a] = {}
            for g, f, t in product(range(self.G), range(self.M), range(self.M)):
                C_sold[a][(g,f,t)] = solver.IntVar(0, infinity, f'C_sold[{a}][{(g,f,t)}]')
                C_sold_seq[a][(g,f,t)] = solver.IntVar(0, infinity, f'C_sold_seq[{a}][{(g,f,t)}]')
                C_sold_par[a][(g,f,t)] = solver.IntVar(0, infinity, f'C_sold_par[{a}][{(g,f,t)}]')

                # partition sold cores into sequential and parallel
                solver.Add(C_sold[a][(g,f,t)] == C_sold_seq[a][(g,f,t)] + C_sold_par[a][(g,f,t)])

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
                self.gamma(g, t) * (self.period_length - self.delta(g, f, t)) * C_sold_par[a][(g,f,t)]
                for g, f, t in product(range(self.G), range(self.M), range(self.M))
            ) == Q_par[a])
            
            # set number of cycles provided (approximate harmonic mean with arithmetic mean)
            solver.Add(sum(
                (1 - A.parallel_fraction) * Q_seq[a] + A.parallel_fraction * Q_par[a]
                for g, f, t in product(range(self.G), range(self.M), range(self.M))
            ) == Q[a])
    
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
                sum(self.CORES * M_sold[(g,f,t)] for f in range(self.M)) == 
                sum(
                    C_sold[a][(g,f,t)] 
                    for a, f in product(applications.keys(), range(self.M))
                ) + C_partunsold[(g,t)]
            )
            solver.Add(
                sum(self.CORES * M_unsold[(g,f,t)] for f in range(self.M)) == 
                sum(C_unsold[(g,f,t)] for f in range(self.M)) - C_partunsold[(g,t)] # CHECK changed + to - here?
            )
        for g in range(self.G):
            solver.Add(
                self.MACHINES == sum(
                    M_sold[(g,f,t)] + M_unsold[(g,f,t)]
                    for f, t in product(range(self.M), range(self.M))
                )
            )
        for g, f in product(range(self.G), range(self.M)): # maintain consistency with current core states
            solver.Add( sum( C_sold[a][(g,f,t)] for a, t in product(applications.keys(), range(self.M))) 
                    + sum( C_unsold[(g,f,t)] for t in range(self.M) )
                    == C_current[(g, f)] )
    
        # seller cost model
        E_sold = solver.NumVar(0, infinity, 'E_sold')
        H_sold = solver.NumVar(0, infinity, 'H_sold')
        E_mult = 2
        E_trans = lambda g, f, t : 0 if f == t else 1 + g
        E_base_active = lambda g, tau, t : 0.1 * t * tau
        E_core_active = lambda g, tau, t : 0.7 * t * tau
        H_trans = lambda g, f, t : 0 if f == t else 1 + g
        H_base_active = lambda g, tau, t : 0.1 * t * tau
        solver.Add(
            E_sold == sum(
                E_mult * (E_trans(g,f,t) + E_base_active(g, self.period_length, t)) * 
                M_sold[(g,f,t)]
                for g, f, t in product(range(self.G), range(self.M), range(self.M)) 
            ) + sum(
                E_mult * E_core_active(g, self.period_length, t) * C_sold[a][(g,f,t)]
                for a, g, f, t in product(applications.keys(), range(self.G), range(self.M), range(self.M)) 
            )
        )
        solver.Add(
            H_sold == sum(
                (H_trans(g,f,t) + H_base_active(g, self.period_length, t)) * M_sold[(g,f,t)]
                for g, f, t in product(range(self.G), range(self.M), range(self.M))
            )
        )
    
        #print('Number of variables =', solver.NumVariables())
        #print('Number of constraints =', solver.NumConstraints())
    
        solver.Maximize(
            sum(V[a] for a in applications.keys()) - E_sold - H_sold
        )
    
        status = solver.Solve()
    
        if status == pywraplp.Solver.OPTIMAL:
            #print('Solution:')
            #print('Objective value =', solver.Objective().Value())
            if save_alloc:
                self.saved_core_allocation = { a : {} for a in applications.keys() }
                self.saved_unsold_cores = {}
                for a, g, t in product(applications.keys(), range(self.G), range(self.M)): # will need to change for core holding
                    self.saved_core_allocation[a][(g,t)] = sum([C_sold[a][(g,f,t)].solution_value() for f in range(self.M)])
                    self.saved_unsold_cores[(g,t)] = sum([C_unsold[(g,f,t)].solution_value() for f in range(self.M)])

            return solver.Objective().Value(), {a : V[a].solution_value() for a in V}
        else:
            print('The problem does not have an optimal solution.')
