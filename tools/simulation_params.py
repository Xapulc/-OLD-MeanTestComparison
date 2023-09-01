class SimulationParams(object):
    def __init__(self, x_sample_size, y_sample_size,
                 sample_name, iter_size, dist_list):
        self.x_sample_size = x_sample_size
        self.y_sample_size = y_sample_size
        self.sample_name = sample_name
        self.iter_size = iter_size
        self.dist_list = dist_list
