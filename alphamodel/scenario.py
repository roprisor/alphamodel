"""
Scenario class
    holds everything required to run a sim in cvxportfolio
"""

__all__ = ['Scenario']


class Scenario:
    """
    Scenario needs per stock: returns, volumes, sigmas
    """
    def __init__(self, returns, volumes, sigmas):
        self.returns = returns
        self.volumes = volumes
        self.sigmas = sigmas
