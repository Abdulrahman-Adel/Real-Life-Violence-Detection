class BaseDataLoader:
    def __init__(self, config):
        self.config = config

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass

    def preprocess(self):
        pass
    
    #for future implementation
    def data_generator(self):
        pass
