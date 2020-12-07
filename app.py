from typing import List, Tuple
import random
from sla import SLA

class App:
    def __init__(self, sla, demand_func=lambda t : random.choice([2, 10, 20, 30, 35, 70]), greedy_param=0):
        self.sla = sla
        self.demand_func = demand_func
        self.current_demand = None

        self.current_cycles = None
        self.current_price = None

    # necessary to maintain demand at a given time when it's randomized
    def set_demand(self, current_time):
        self.current_demand = self.demand_func(current_time)

    def will_hold(self, next_time):
        if self.current_cycles == None:
            return False

        TRIALS = 100
        expected_demand = 0
        for i in range(TRIALS):
            expected_demand += self.demand_func(next_time)
        expected_demand /= TRIALS

        surplus = sla.eval_exact_value(self.current_cycles, expected_demand) - self.current_price
        return surplus >= greedy_param
