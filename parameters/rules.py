class Rules(object):
    def __init__(self, ncells: int, activation_interval: range, distance: int, periodic: bool) -> None:
        self.ncells = ncells
        self.activation_interval = activation_interval
        self.distance = distance
        self.periodic = periodic
