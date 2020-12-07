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
applications = {
    app_id : App(SLA(endpoints[sla_ids[app_id]])) for app_id in range(len(sla_ids))
}

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
    market = Market(applications, G, M, CORES, MACHINES, gamma, delta, 10)
    for i in range(2):
        print(market.get_allocation_and_prices(False, True))
        for app_id, app in applications.items():
            print(app_id, app.current_cycles, app.current_price)        
        market.advance_time()

