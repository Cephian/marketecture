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
import numpy as np
import random

#############################################
# Input Data
#############################################

# endpoints encoding SLAs
endpoints = [
    [
        (0, 5), (0.05, 5), (0.1, 4.5), (0.5, 2), (1, 0)
    ],
    [
        (0, 4.5), (0.05, 4.5), (0.1, 3.5), (0.5, 1.5), (1, 0)
    ],
]
# array of SLA ids
sla_ids = [
    0, 0, 1, 1,
]

# demand function from time to transactions/min demanded
df = lambda t : random.choice([2, 10, 20, 30, 35, 70])

# array of applications
applications = {
        app_id : App(SLA(endpoints[sla_ids[app_id]]), df) for app_id in range(len(sla_ids))
}

if __name__ == '__main__':
    market = Market(applications)
    for i in range(2):
        print(market.get_allocation_and_prices(False, True))
        for app_id, app in applications.items():
            print(app_id, app.current_cycles, app.current_price)        

