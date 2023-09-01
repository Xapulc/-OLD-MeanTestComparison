import numpy as np
import pandas as pd

from tqdm import tqdm
from statsmodels.stats.weightstats import ztest
from scipy.stats import bootstrap, ttest_ind, permutation_test

from .util import load_file, save_file


class Simulation(object):
    def __init__(self, simulation_param_list, root, title, subtitle,
                 alpha_list, alternative, true_hypothesis):
        self.simulation_param_list = simulation_param_list
        self.root = root
        self.title = title
        self.subtitle = subtitle
        self.alpha_list = alpha_list
        self.alternative = alternative
        self.true_hypothesis = true_hypothesis

    def _test_list(self):
        permutation_n_resamples = lambda x, y: 9999
        permutation_batch = lambda x, y: int(permutation_n_resamples(x, y) * min(1.0, 10000 / max(len(x), len(y))))

        bootstrap_n_resamples = lambda x, y: 1000
        bootstrap_batch = lambda x, y: int(bootstrap_n_resamples(x, y) * min(1.0, 100000 / max(len(x), len(y))))

        test_list = [{
            "name": "z-test",
            "pvalue": lambda x, y, *args, **kwargs: ztest(x, y, alternative=self.alternative)[1]
        }, {
            "name": "t-test, equal var = true",
            "pvalue": lambda x, y, *args, **kwargs: ttest_ind(x, y,
                                                              alternative=self.alternative,
                                                              equal_var=True).pvalue
        }, {
            "name": "t-test, equal var = false",
            "pvalue": lambda x, y, *args, **kwargs: ttest_ind(x, y,
                                                              alternative=self.alternative,
                                                              equal_var=False).pvalue
        }, {
            "name": "permutation test, mean stat",
            "pvalue": lambda x, y, random_state, *args, **kwargs: permutation_test((x, y),
                                                                                   lambda x_, y_, axis: np.mean(x_, axis=axis) - np.mean(y_, axis=axis),
                                                                                   vectorized=True,
                                                                                   random_state=random_state+1,
                                                                                   n_resamples=permutation_n_resamples(x, y),
                                                                                   batch=permutation_batch(x, y),
                                                                                   alternative=self.alternative).pvalue
        }, {
            "name": "permutation test, t-test stat",
            "pvalue": lambda x, y, random_state, *args, **kwargs: permutation_test((x, y),
                                                                                   lambda x_, y_, axis: ttest_ind(x_, y_,
                                                                                                                  axis=axis,
                                                                                                                  alternative=self.alternative,
                                                                                                                  equal_var=False).statistic,
                                                                                   vectorized=True,
                                                                                   random_state=random_state+10,
                                                                                   n_resamples=permutation_n_resamples(x, y),
                                                                                   batch=permutation_batch(x, y),
                                                                                   alternative=self.alternative).pvalue
        }, {
            "name": "t-permutation test, equal var = false",
            "pvalue": lambda x, y, random_state, *args, **kwargs: ttest_ind(x, y,
                                                                            permutations=permutation_n_resamples(x, y),
                                                                            alternative=self.alternative,
                                                                            random_state=random_state+11,
                                                                            equal_var=False).pvalue
        }, {
            "name": "bootstrap, percentile method",
            "bootstrap_result": lambda x, y, random_state, alpha, bootstrap_result=None, *args, **kwargs: bootstrap([x, y],
                                                                                                                    lambda x_, y_, axis: np.mean(x_, axis=axis) - np.mean(y_, axis=axis),
                                                                                                                    method="percentile",
                                                                                                                    random_state=random_state+6,
                                                                                                                    confidence_level=1-alpha,
                                                                                                                    alternative=self.alternative,
                                                                                                                    bootstrap_result=bootstrap_result,
                                                                                                                    n_resamples=bootstrap_n_resamples(x, y),
                                                                                                                    batch=bootstrap_batch(x, y),
                                                                                                                    paired=False)
        }, {
            "name": "bootstrap, basic method",
            "bootstrap_result": lambda x, y, random_state, alpha, bootstrap_result=None, *args, **kwargs: bootstrap([x, y],
                                                                                                                    lambda x_, y_, axis: np.mean(x_, axis=axis) - np.mean(y_, axis=axis),
                                                                                                                    method="basic",
                                                                                                                    random_state=random_state+7,
                                                                                                                    confidence_level=1-alpha,
                                                                                                                    alternative=self.alternative,
                                                                                                                    bootstrap_result=bootstrap_result,
                                                                                                                    n_resamples=bootstrap_n_resamples(x, y),
                                                                                                                    batch=bootstrap_batch(x, y),
                                                                                                                    paired=False)
        }, {
            "name": "bootstrap, bca method",
            "bootstrap_result": lambda x, y, random_state, alpha, bootstrap_result=None, *args, **kwargs: bootstrap([x, y],
                                                                                                                    lambda x_, y_, axis: np.mean(x_, axis=axis) - np.mean(y_, axis=axis),
                                                                                                                    method="bca",
                                                                                                                    random_state=random_state+8,
                                                                                                                    confidence_level=1-alpha,
                                                                                                                    alternative=self.alternative,
                                                                                                                    bootstrap_result=bootstrap_result,
                                                                                                                    n_resamples=bootstrap_n_resamples(x, y),
                                                                                                                    batch=bootstrap_batch(x, y),
                                                                                                                    paired=False)
        }]

        return test_list

    def _test_list_result(self, x, y, random_state, test_name_list=None):
        default_test_list = self._test_list()
        if test_name_list is None:
            test_list = default_test_list
        else:
            test_list = list(filter(lambda test: test["name"] in test_name_list,
                                    default_test_list))

        bootstrap_result = None
        for test in test_list:
            test["result"] = {}

            for alpha in self.alpha_list:
                if "pvalue" in test.keys():
                    p_value = test["pvalue"](x, y, random_state=random_state)
                    if p_value > alpha:
                        test["result"][alpha] = 0
                    else:
                        test["result"][alpha] = 1
                else:
                    bootstrap_result = test["bootstrap_result"](x, y, random_state=random_state,
                                                                alpha=alpha, bootstrap_result=bootstrap_result)

                    if self.alternative == "less":
                        if 0 <= bootstrap_result.confidence_interval.high:
                            test["result"][alpha] = 0
                        else:
                            test["result"][alpha] = 1
                    elif self.alternative == "greater":
                        if bootstrap_result.confidence_interval.low <= 0:
                            test["result"][alpha] = 0
                        else:
                            test["result"][alpha] = 1
                    else:
                        if bootstrap_result.confidence_interval.low <= 0 <= bootstrap_result.confidence_interval.high:
                            test["result"][alpha] = 0
                        else:
                            test["result"][alpha] = 1

        res = [{
            "name": test["name"],
            "result": test["result"]
        } for test in test_list]

        return res

    def start(self, random_state=None, rewrite_result=False, result_disable=True, tqdm_disable=True):
        for simulation_params in self.simulation_param_list:
            for dist_couple in simulation_params.dist_list:
                name = f"{simulation_params.sample_name}; {dist_couple.dist_name}"
                all_data = load_file(self.root, self.title, self.subtitle, name + ".csv")

                test_name_list = [test["name"] for test in self._test_list()]
                if all_data is not None and not rewrite_result:
                    test_name_list = list(filter(lambda test_name: test_name not in all_data.index.unique(),
                                                 test_name_list))
                else:
                    test_name_list = test_name_list

                if len(test_name_list) > 0:
                    x = dist_couple.x_dist.rvs(size=[simulation_params.iter_size,
                                                     simulation_params.x_sample_size],
                                               random_state=random_state-1 if random_state is not None else None)
                    y = dist_couple.y_dist.rvs(size=[simulation_params.iter_size,
                                                     simulation_params.y_sample_size],
                                               random_state=random_state-2 if random_state is not None else None)

                    for i in tqdm(range(simulation_params.iter_size), disable=tqdm_disable):
                        test_result = self._test_list_result(x[i], y[i], random_state,
                                                             test_name_list=test_name_list)

                        test_result_data = pd.DataFrame([{
                            "name": test["name"],
                            "iter_num": i,
                            **test["result"]
                        } for test in test_result]) \
                            .set_index(["name", "iter_num"]) \
                            [self.alpha_list]

                        if all_data is None:
                            all_data = test_result_data
                        else:
                            all_data = pd.concat([all_data, test_result_data])
                else:
                    all_data = all_data.rename(columns={
                        f"{alpha}": alpha
                        for alpha in self.alpha_list
                    })

                if len(test_name_list) > 0:
                    save_file(all_data, self.root, self.title, self.subtitle, name + ".csv")

                if not result_disable:
                    test_stat = all_data.groupby("name") \
                                        .agg({alpha: "mean" for alpha in self.alpha_list}) \
                                        .sort_index()

                    print("-------")

                    if self.true_hypothesis == 0:
                        print(f"{name}, FPR")
                        print(test_stat)
                    else:
                        print(f"{name}, FNR")
                        print(test_stat * (-1) + 1)

                    print("-------")
