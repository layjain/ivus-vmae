import wandb


class Visualize(object):
    def __init__(self, args):
        self.args = args
        self._init = False

    def wandb_init(self, model):
        if not self._init:
            self._init = True
            wandb.init(project="videowalk", group="release", config=self.args)
            wandb.watch(model)

    def log(self, key_vals):
        return wandb.log(key_vals)
