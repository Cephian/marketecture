from typing import List, Tuple
import random
from sla import SLA

class App:
    def __init__(self, sla, demand_func=lambda t : random.choice([2, 10, 20, 30, 35, 70])):
        self.sla = sla
        self.demand_func = demand_func
        self.current_demand = None

    # necessary to maintain demand at a given time when it's randomized
    def set_demand(self, current_time):
        self.current_demand = self.demand_func(current_time)
