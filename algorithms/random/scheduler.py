class RandomScheduler:
    def __init__(self, seed):
        self.seed = seed

    def select_vm(self, vms):
        return self.seed.choice(vms)