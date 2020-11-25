from typing import List, Tuple

from sla import SLA

class App:
    def __init__(self, sla):
        self.sla = sla
        self.demand_estimate = 0