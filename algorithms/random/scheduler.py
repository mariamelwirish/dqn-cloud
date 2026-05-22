from algorithms.base_scheduler import BaseScheduler

class RandomScheduler(BaseScheduler):
    def __init__(self, seed):
        self.seed = seed
    
    def select_vm(self, vms, task):
        vm = self.seed.integers(0, len(vms))
        return vms[vm]