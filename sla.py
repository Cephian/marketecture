import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class SLA:
    '''
    Piecewise linear function representing the value for the Pth
    percentile response time.
    '''
    P = 0.95
    TAU = 10 # 10 minutes
    MONTH = 60 * 24 * 30 # minutes in 30 days

    def __init__(self, endpoints: List[Tuple[float, float]]):
        x = [p[0] for p in endpoints]
        y = [p[1] for p in endpoints]

        assert all(xi < xj for xi, xj in zip(x[:-1], x[1:])), \
            'Endpoints do not have strictly increasing x coordinates.'
        assert all(yi >= yj for yi, yj in zip(y[:-1], y[1:])), \
            'Endpoints do not have decreasing y coordinates.'
        assert x[0] == 0, 'First coordinate must have x = 0.'
        assert y[-1] == 0, 'Endpoints do not end at y = 0.'

        self.endpoints = endpoints
        self.x = x
        self.y = y

    @staticmethod
    def eval_piecewise(x, y, t):
        i = np.searchsorted(x, t)
        if i >= len(x):
            return 0
        if i == 0:
            return y[0]
        m = (y[i] - y[i-1]) / (x[i] - x[i-1])

        return y[i-1] + m * (t - x[i-1])

    def eval_sla(self, response_time: float):
        return SLA.eval_piecewise(self.x, self.y, response_time)

    def eval_exact_value(self, supply: float, demand: float):
        '''
        Value without piecewise linear approximation
        '''
        if supply <= demand:
            return 0
        q = -np.log(1 - SLA.P) / (supply - demand)
        return self.eval_sla(q)

    def eval_value(self, supply: float, demand: float):
        '''
        Value with piecewise linear approximation
        '''
        if supply <= demand:
            return 0
        # supply mu such that x = -log(1-P) / (mu - lambda) for control points
        self.supply_x = [(-np.log(1 - SLA.P) + xi * demand) / xi for xi in self.x]
        self.value_y = [self.eval_exact_value(s, demand) for s in self.supply_x]
        return SLA.eval_piecewise(self.supply_x, self.value_y, supply)

    def plot_sla(self, filename: str):
        a, b = 0, max(self.x)
        x = np.arange(a, b, 1e-2)
        y = np.vectorize(lambda y : self.eval_sla(y))(x)
        plt.plot(x, y)
        plt.title(f'Value for {SLA.P * 100}th Percentile Response Time')
        plt.xlabel('Response Time per Transaction in Seconds')
        plt.ylabel('Value per Month ($)')
        plt.scatter(self.x, self.y)
        plt.savefig(filename)
        plt.clf()

    def plot_supply_value(self, filename: str, demand: float):
        a, b = 0, 80

        self.supply_x = [
            (-np.log(1 - SLA.P) + xi * demand) / xi for xi in self.x
        ][::-1]
        self.value_y = [
            self.eval_exact_value(s, demand) * SLA.TAU / SLA.MONTH
            for s in self.supply_x
        ]

        x = np.arange(a, b, 1e-2)
        y = np.vectorize(lambda mu : SLA.eval_piecewise(self.supply_x, self.value_y, mu))(x)
        print('supply', self.supply_x)
        print('value', self.value_y)
        plt.plot(x, y)
        plt.title(f'Supply Value Cruve')
        plt.xlabel('Supply in TCycles / Period')
        plt.ylabel('$ / Period')
        plt.scatter(self.supply_x, self.value_y)
        plt.savefig(filename)
        plt.clf()

if __name__ == '__main__':
    endpoints1 = [
        (0, 50000), (0.05, 50000), (0.1, 45000), (0.5, 20000), (1, 0)
    ]
    endpoints2 = [
        (0, 45000), (0.05, 45000), (0.1, 35000), (0.5, 15000), (1, 0)
    ]

    sla1 = SLA(endpoints1)
    sla2 = SLA(endpoints2)

    sla1.plot_sla('sla1.pdf')
    sla2.plot_sla('sla2.pdf')
    sla1.plot_supply_value('supply_value1.pdf', 10)
    sla2.plot_supply_value('supply_value2.pdf', 10)
