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
from math import atan
import numpy as np
import random

#############################################
# Input Data
#############################################

# number of applications/users
num_apps = 5

# number of allocationrounds to run
num_rounds = 8

# different unscaled value functions
def f1(x):
    if x < 0.000001:
        x = 0.000001
    return 1.0/x-1

def f2(x):
    if x < 0.000001:
        x = 0.000001
    return -log(x)

def f3(x):
    return 1-x**2

def f3(x):
    return (atan(4*(1-x)-2) / atan(2))+1

def rand_sla():
	c = random.uniform(0.5, 2)
	L = [x*0.1 for x in range(11)] + [0.25*2**(-x) for x in range(10)]
	L = sorted(set(L))
	return [(x, f3(x) * c) for x in L]

# demand function from time to transactions/min demanded
df = lambda t : random.randrange(00000, 50000)

# array of applications
applications = {
		app_id : App(SLA(rand_sla()), df) for app_id in range(num_apps)
}

if __name__ == '__main__':
    market = Market(applications)
    for i in range(num_rounds):
        print('--- round ', i,'---')
        print()
        total_welfare, total_values, VCG = market.get_allocation_and_prices(False, False)
        print('total welfare:', total_welfare)
        for app_id, app in applications.items():
            print('application', app_id)
            print('  cycles:', app.current_cycles[0])
            print('  VCG price:', VCG[app_id])
            print('  value:', total_values[app_id])
        print()


