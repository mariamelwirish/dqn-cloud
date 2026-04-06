class EarliestVMScheduler:
    def select_vm(self, vms):
        # Find the VM with the earliest available time
        earliest_vm = min(vms, key=lambda vm: vm.available_time)
        return earliest_vm