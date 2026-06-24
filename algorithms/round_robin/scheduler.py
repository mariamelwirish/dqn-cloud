from algorithms.base_scheduler import BaseScheduler

class RoundRobinScheduler(BaseScheduler):
    def __init__(self):
        self.current_index = 0
    
    def select_vm(self, vms, task):
        self.current_index = self.current_index % len(vms)
        vm = vms[self.current_index]
        self.current_index += 1
        return vm