import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, auc
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

"""
    _find_ppm_sheet(filname, ppm)

read in excel file and check the sheet names which include the ppm number.
"""

def _find_ppm_sheet(filename):
    xlfl = pd.ExcelFile(filename)
    sheet_names = xlfl.sheet_names
    target_sheet = sheet_names[0]
    return target_sheet

"""
    _find_header_row(filename, search_terms=['Time', 's'])

read in excel file and check first ten rows for search terms.
return the first row in which a search term appears.
if not found, return None.
"""
def _find_header_row(filename, ppm_sheet=0, search_terms=['Time', 's']):
    for i in range(10):  # Check first 10 rows
        try:
            df = pd.read_excel(filename, sheet_name=ppm_sheet, header=i, nrows=1)
            for search_term in search_terms:
                if search_term in df.columns:
                    return i
        except:
            pass
    return None  # If header not found

"""
    linear_regression(df, ids_split)

perform linear regression on df[ids_split]:
ΔG/G0 = m * t + b

# arguments:
* df := dataframe of a single partition of sensor_response data
* ids_split := indices of response data partition

# output: dict of:
* coef := coefficient from linear regression
* r2 := r2 score
* ids_split
"""
def linear_regression(df, ids_split):
    X = df.loc[ids_split, "s"].to_numpy().reshape(-1, 1)
    y = df.loc[ids_split, "-ΔG/G0"].to_numpy()

    reg = LinearRegression().fit(X, y)

    r2 = reg.score(X, y)

    slope = reg.coef_[0]
    intercept = reg.intercept_

    return {'slope': slope, 'r2': r2, 'ids_split': ids_split, 'intercept': intercept}

class SensorResponse:
    def __init__(self, data, plot_title):
        self.data = data
        self.title = plot_title
        # store features
        self.slope_info = None
        self.saturation = None
        self.auc = None

    """
    compute_initial_slope(self, partition_size, total_time_window, mse_bound)
    estimate initial slope of data.

      arguments:
          * max_time := indicates the window of time from 0 to max_time to partition data
          * n_partitions := number of partitions
          * r2_bound := bound on acceptable r-squared values from linear regression
    """
    def compute_initial_slope(self, n_partitions=15, max_time=750.0, r2_bound=0):
        early_df = self.data[self.data["s"] < max_time]

        # partition data indices
        ids_splits = np.array_split(early_df.index, n_partitions)

        # create list of regression on each partition of data which satisfy the mean_squared error bound
        regression_data = [linear_regression(early_df, ids_split) for ids_split in ids_splits]
        # filter according to r2
        regression_data = list(filter(lambda res: res['r2'] > r2_bound, regression_data))

        if len(regression_data) == 0:
            raise Exception("Data has no initial slopes that satisfy r2 bound.")

        # find index of max absolute value of linear regression coefficients
        id_initial_slope = np.argmax([np.abs(rd['slope']) for rd in regression_data])

        # return regression_data which contains the initial slope
        self.slope_info = regression_data[id_initial_slope]
        return self.slope_info

    def compute_saturation(self, n_partitions=50):
        ids_splits = np.array_split(self.data.index, n_partitions)

        # get mean over partitions
        means = [np.mean(self.data.iloc[ids_split]['-ΔG/G0']) for ids_split in ids_splits]
        id_max_magnitude = np.argmax(np.abs(means))

        self.saturation = means[id_max_magnitude]
        return self.saturation

    # compute area under curve for each GBx DeltaG/G0 using sklearn auc
    def compute_area_under_response_curve(self):
        self.auc = auc(self.data["s"], self.data['-ΔG/G0'])
        return self.auc

    def compute_features(self, n_partitions_saturation=100, n_partitions_slope=15, r2_bound_slope=0):
        self.compute_saturation(n_partitions=n_partitions_saturation)
        self.compute_initial_slope(n_partitions=n_partitions_slope, r2_bound=r2_bound_slope)
        self.compute_area_under_response_curve()

    def viz(self, save=True): # viz the data along with the response features or function u fit to it.
        if self.slope_info == None or self.saturation == None:
            raise Exception("Compute features first.")

        fig, ax = plt.subplots()

        plt.xlabel("time [s]")
        plt.ylabel(r"$\Delta G/G_0$")

        # plot raw response data
        plt.scatter(self.data['s'], self.data['-ΔG/G0'])

        ###
        #   viz features
        ###
        # saturation
        plt.axhline(self.saturation, linestyle='-', color="gray")

        # slope
        t_start = self.data.loc[self.slope_info["ids_split"][0], 's']
        t_end = self.data.loc[self.slope_info["ids_split"][-1], 's']
        plt.plot(
            [t_start, t_end],
            self.slope_info["slope"] * np.array([t_start, t_end]) + self.slope_info["intercept"],
            color='orange'
        )

        all_info = self.title
        plt.title(all_info)

        if save:
            plt.savefig("responses/featurized_{}.png".format(all_info), format="png")
        plt.show()
