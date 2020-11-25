'''
Implementation of market mechanism in

Expressive Power-Based Resource Allocation for Data Centers

https://www.ijcai.org/Proceedings/09/Papers/243.pdf

For simplicity, we make the following assumptions for now:
* Machine mode cycles gamma_{g, t}(Delta) is proportional to Delta
* Bidding proxies estimate demand with a single number rather than averaging

OR-Tools tutorial:
https://developers.google.com/optimization/mip/integer_opt#define-the-variables

MIP formulation of piecewise linear functions:
http://yetanothermathprogrammingconsultant.blogspot.com/2015/10/piecewise-linear-functions-in-mip-models.html
'''

import numpy as np
import random
from itertools import product
from ortools.linear_solver import pywraplp

from sla import SLA
from app import App

#############################################
# Input Data
#############################################

# endpoints encoding SLAs
endpoints = [
    [
        (0, 50000), (0.05, 50000), (0.1, 45000), (0.5, 20000), (1, 0)
    ],
    [
        (0, 45000), (0.05, 45000), (0.1, 35000), (0.5, 15000), (1, 0)
    ],
]
# array of SLA ids
sla_ids = [
    0, 0, 1, 1,
]
# array of applications
applications = [
    App(SLA(endpoints[i])) for i in sla_ids
]

G = 2 # machine groups
M = 3 # power modes
# TCycles per minute for machine groups and power modes
gamma = lambda g, t : [
    [0.5, 1, 1.5],
    [1, 2, 3],
][g][t]
# minutes to switch power modes
delta = lambda g, f, t : 0 if f == t else 1 + g
CORES = 2
MACHINES = 2


if __name__ == '__main__':
    solver = pywraplp.Solver.CreateSolver('SCIP')
    infinity = solver.infinity()

    #############################################
    # Define MIP
    #############################################

    # buyer valuation model
    V = {} # value provided to application a V = F(Q)
    Q = {} # cycles provided to application a
    Z_seg = {} # segment indicator for piecewise linear function
    Q_seg = {} # segment value for piecewise linear function
    C_sold = {}
    C_unsold = {}
    C_partunsold = {}
    for a, A in enumerate(applications):
        V[a] = solver.NumVar(0, infinity, f'V[{a}]')
        Q[a] = solver.NumVar(0, infinity, f'Q[{a}]')

        # set measured application demand for A
        A.demand_estimate = random.choice([2, 10, 20, 30, 35, 70])
        A.sla.compute_supply_value(A.demand_estimate)

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
        for g, f, t in product(range(G), range(M), range(M)):
            C_sold[a][(g,f,t)] = solver.IntVar(0, infinity, f'C_sold[{a}][{(g,f,t)}]')
        solver.Add(sum(
            gamma(g, t) * (SLA.TAU - delta(g, f, t)) * C_sold[a][(g,f,t)]
            for g, f, t in product(range(G), range(M), range(M))
        ) == Q[a])

    # goods in the market
    M_sold = {}
    M_unsold = {}
    for g, f, t in product(range(G), range(M), range(M)):
        M_sold[(g,f,t)] = solver.IntVar(0, infinity, f'M_sold[{(g,f,t)}]')
        M_unsold[(g,f,t)] = solver.IntVar(0, infinity, f'M_unsold[{(g,f,t)}]')
        C_unsold[(g,f,t)] = solver.IntVar(0, infinity, f'C_unsold[{(g,f,t)}]')
    for g, t in product(range(G), range(M)):
        C_partunsold[(g,t)] = solver.IntVar(0, infinity, f'C_partunsold[{(g,t)}]')

    for g, t in product(range(G), range(M)):
        solver.Add(
            sum(CORES * M_sold[(g,f,t)] for f in range(M)) == 
            sum(
                C_sold[a][(g,f,t)] 
                for a, f in product(range(len(applications)), range(M))
            ) + C_partunsold[(g,t)]
        )
        solver.Add(
            sum(CORES * M_unsold[(g,f,t)] for f in range(M)) == 
            sum(C_unsold[(g,f,t)] for f in range(M)) + C_partunsold[(g,t)]
        )
    for g in range(G):
        solver.Add(
            MACHINES == sum(
                M_sold[(g,f,t)] + M_unsold[(g,f,t)]
                for f, t in product(range(M), range(M))
            )
        )

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
            E_mult * (E_trans(g,f,t) + E_base_active(g,SLA.TAU,t)) * 
            M_sold[(g,f,t)]
            for g, f, t in product(range(G), range(M), range(M)) 
        ) + sum(
            E_mult * E_core_active(g,SLA.TAU,t) * C_sold[a][(g,f,t)]
            for a, g, f, t in product(range(len(applications)), range(G), range(M), range(M)) 
        )
    )
    solver.Add(
        H_sold == sum(
            (H_trans(g,f,t) + H_base_active(g,SLA.TAU,t)) * M_sold[(g,f,t)]
            for g, f, t in product(range(G), range(M), range(M))
        )
    )

    print('Number of variables =', solver.NumVariables())
    print('Number of constraints =', solver.NumConstraints())

    solver.Maximize(
        sum(V[a] for a in range(len(applications))) - E_sold - H_sold
    )

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
    else:
        print('The problem does not have an optimal solution.')