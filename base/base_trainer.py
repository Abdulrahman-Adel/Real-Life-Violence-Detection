class BaseTrain(object):
    def __init__(self, model, data_train,data_validate,config):
        self.model = model
        self.train_data = data_train
        self.val_data = data_validate
        self.config = config

    def train(self):
        raise NotImplementedError
