from algorithms.base_scheduler import BaseScheduler

class EarliestVMScheduler(BaseScheduler):
    def select_vm(self, vms, task):
        vm = min(vms, key=lambda vm: vm.next_available_time)
        return vm