from typing import List, Tuple
import random
from sla import SLA

class App:
    def __init__(self, sla, demand_func, greedy_param=float('inf'), parallel_func=lambda t:1):
        self.sla = sla # units as described in sla.py
        '''
        Our  realistic  but  synthetic  traces  are  the  sum  of  two  si-nusoid  curves  (e.g.   
        1  day  period  with  9000  peak  transac-tions/minute  plus  1  week  period  with  36000  peak  
        transac-tions/minute) and a noise term drawn from a Gaussian with as.d.  equal to 25% of the signal.
        '''
        self.demand_func = demand_func # function: current time -> demand (transactions / minute)
        self.greedy_param = greedy_param # required surplus to hold ($)
        self.parallel_func = parallel_func # current time -> [0, 1] requested fraction of the period on which sequential cycles are used

        self.current_demand = None
        self.parallel_fraction = None

        # the allocation and price the user got in the last market period
        self.current_cycles = None # MCycles
        self.current_price = None # $

    # necessary to maintain demand at a given time when it's randomized
    def set_demand(self, current_time):
        self.current_demand = self.demand_func(current_time)
        self.parallel_fraction = self.parallel_func(current_time)

    # asks if the user will hold the number of cycles they currently recieve 
    # (from the last allocation) for the period starting at current_time
    def will_hold(self, current_time, period_length):
        if self.current_cycles == None:
            return False

        # users have to choose whether to hold before seeing this period's demand/frac, 
        # so estimate next period value with current alloc
        TRIALS = 100
        v = 0
        for i in range(TRIALS):
            d = self.demand_func(current_time)
            f = self.parallel_func(current_time)
            q = ((1-f) * self.current_cycles[1]) + (f * self.current_cycles[2]) 
            v += self.sla.eval_exact_value(q, d) * (period_length / SLA.TIME_UNIT)
        v /= TRIALS

        surplus = v - self.current_price
        return surplus >= self.greedy_param
