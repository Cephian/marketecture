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

from sla import SLA
from app import App
from market import Market
from math import log
import numpy as np
import random

#############################################
# Input Data
#############################################

def f1(x):
    if x < 0.000001:
        x = 0.000001
    return 1.0/x-1

def f2(x):
    if x < 0.000001:
        x = 0.000001
    return -log(x)

def f3(x):
    if x < 0.000001:
        x = 0.000001
    return 1-x**2

# endpoints encoding SLAs
#endpoints = [
#    [
#        (0, 100), (0.00001, 100), (0.5, 5), (1, 0)
#    ],
#    [
#        (0, 50), (0.00001, 50), (0.1, 0),
#    ],
#]

L = [0, 0.00001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
endpoints = [
    [
        (x, f1(x)) for x in L
    ],
    [
        
        (x, f2(x)) for x in L
    ],
    [
        
        (x, f3(x)) for x in L
    ],
    [
        
        (x, f1(x)**2) for x in L
    ],
]

# array of SLA ids
sla_ids = [
    2, 2, 2, 2
]

# demand function from time to transactions/min demanded
df = lambda t: 1000*t#lambda t : random.choice([2, 10, 20, 30, 35, 70])
# so each period 30000 MCYcles at least needed

# array of applications
applications = {
        app_id : App(SLA(endpoints[sla_ids[app_id]]), df) for app_id in range(len(sla_ids))
}

if __name__ == '__main__':
    market = Market(applications)
    for i in range(5):
        total_welfare, total_values, VCG = market.get_allocation_and_prices(False, False)
        #print(market.get_allocation_and_prices(False, False))
        #print('allocation:', market.current_core_allocation, market.current_unsold_cores)
        print()
        for app_id, app in applications.items():
            print(app_id, app.current_cycles, app.current_price)        


