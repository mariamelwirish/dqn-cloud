class RoundRobinScheduler:
    def __init__(self):
        self.current_vm_index = 0

    def select_vm(self, vms):
        vm = vms[self.current_vm_index]
        self.current_vm_index = (self.current_vm_index + 1) % len(vms)
        return vm