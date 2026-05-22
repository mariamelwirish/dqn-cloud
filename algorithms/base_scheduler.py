'''
    Abstract base class for all scheduling algorithms.
    All schedulers must implement the select_vm method.
'''

class BaseScheduler:
    def select_vm(self, vms, task):
        raise NotImplementedError