from scipy.stats import norm


class DistributionCouple(object):
    def __init__(self, x_dist_family, y_dist_family, dist_name=None,
                 x_dist_param_dict=None, y_dist_param_dict=None):
        if x_dist_param_dict is not None:
            self.x_dist = x_dist_family(**x_dist_param_dict)
        else:
            self.x_dist = x_dist_family

        if y_dist_param_dict is not None:
            self.y_dist = y_dist_family(**y_dist_param_dict)
        else:
            self.y_dist = y_dist_family

        if dist_name is not None:
            self.dist_name = dist_name
        else:
            x_dist_name = self._get_dist_name(x_dist_family, x_dist_param_dict)
            y_dist_name = self._get_dist_name(y_dist_family, y_dist_param_dict)

            self.dist_name = f"{x_dist_name} vs {y_dist_name}"

    def _get_dist_name(self, dist_family, dist_param_dict):
        if dist_family.name == norm.name:
            return self._get_norm_dist_name(dist_param_dict)
        else:
            raise ValueError(f"Incorrect distribution family: {dist_family}")

    def _get_norm_dist_name(self, dist_param_dict):
        if dist_param_dict is not None:
            assert "loc" in dist_param_dict.keys() and "scale" in dist_param_dict.keys(), \
                f"Incorrect params dict: {dist_param_dict}"

            loc = dist_param_dict["loc"]
            var = dist_param_dict["scale"]**2
        else:
            loc = norm.mean()
            var = norm.var()

        return f"N({loc}, {var})"
