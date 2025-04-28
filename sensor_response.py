import marimo

__generated_with = "0.10.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from pathlib import Path

    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    import seaborn as sns

    from sklearn.preprocessing import PowerTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, auc, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    return (
        Line2D,
        LinearRegression,
        LinearSegmentedColormap,
        PCA,
        Path,
        PowerTransformer,
        RandomForestClassifier,
        auc,
        confusion_matrix,
        make_axes_locatable,
        mean_squared_error,
        np,
        path_effects,
        pd,
        plt,
        r2_score,
        sns,
    )


@app.cell
def _(plt):
    my_colors = plt.get_cmap('tab10')
    my_colors
    return (my_colors,)


@app.cell
def _(my_colors):
    target_to_shape = {"--" : "o",
                       "+-" : "s",
                       "-+" : "*",
                       "++" : "^"}
    target_to_color = {"--" : my_colors(0),
                       "+-" : my_colors(1),
                       "-+" : my_colors(2),
                       "++" : my_colors(4)}
    return target_to_color, target_to_shape


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Helper function to extract response data from excel file""")
    return


@app.cell
def _(Path, np, pd):
    """
        _find_ppm_sheet(filname, ppm)

    read in excel file and check the sheet names which include the ppm number.
    """

    def _find_ppm_sheet(filename, ppm):
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
        read_data(MOF, ppm)

    read in the sensor response data for a given MOF exposed to a
    given gas mixture concentration.
    returns list of pandas data frames with this data. (may be replicates)
    each data frame has two columns: time, DeltaG/G0.

    note: this is complicated because there are two formats for a given
    cof, gas, carrier, ppm:
    (1) multiple replicates in the same file
    (2) multiple replicates in separate files
    """
    def read_data(MOF, ppm, time_adjust=0):
        ppms = ["0+5", "0+10", "0+20", "0+40", "0+80", "5+0", "10+0", "20+0", "40+0", "80+0", "4+36", "5+5", "9+13", 
            "13+27", "18+4", "22+18", "27+31", "31+9","36+22", "40+40"]

        path = Path.cwd().joinpath("H2S+SO2-data", f"{ppm}ppm", MOF).rglob("*.xlsx")

        # folders contain multiple excel files, so extract relevant
        files = [file for file in path]
        if len(files) > 1:
            raise Exception(f"multiple data in folder {path}")

        # extract data from Excel files in list
        dfs = []
        for filename in files:
            ppm_sheet = None
            if ppm in ppms:
                ppm_sheet = _find_ppm_sheet(filename, ppm)
                # read in file (need to find header row; not consistent)
                header_row = _find_header_row(filename, ppm_sheet)
                df = pd.read_excel(filename, sheet_name=ppm_sheet, header=header_row)

            else:
                raise Exception("PPM not supported.")

            #    only keep a subset of the cols (Time and (perhaps multiple) with muA's)
            ids_cols_keep = df.columns.str.contains('A', na=False) | (df.columns == 's')
            # exposure time begins at 780s, ends 2580s
            start_index = df.index[df['s'] == 780 + time_adjust].tolist()[0]
            end_index = df.index[df['s'] == 2580 + time_adjust].tolist()[0]
            df = df.loc[start_index:end_index, df.columns[ids_cols_keep]]

            # check time is sliced properly
            assert df.iloc[0]["s"] == 780.0 + time_adjust
            assert df.iloc[-1]["s"] == 2580.0 + time_adjust
            # reshift time
            df["s"] = df["s"] - (780.0 + time_adjust)


            # drop columns with missing values
            df = df.dropna(axis='columns')

            df.reset_index(drop=True, inplace=True)

            # separate replicates into differente dataframes and append to dfs
            for i in df.columns:
                if 'A' in i and not np.all(df[i] == 0):
                    data_rep = df[['s', i]]
                    G0 = df[i].iloc[0]
                    # replace muA column with -deltaG/G0 calculation: -ΔG/G0 = -(muA - G0)/G0 * 100
                    data_rep.loc[:, i] = 100 * (-(data_rep.loc[:, i] - G0) / G0)
                    data_rep = data_rep.rename(columns={i: "-ΔG/G0"})
                    dfs.append(data_rep)
        return dfs
    return (read_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Helper function to run linear regression""")
    return


@app.cell
def _(LinearRegression):
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
    return (linear_regression,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Class to extract response features""")
    return


@app.cell
def _(auc, linear_regression, np, plt, read_data):
    class SensorResponse:
        def __init__(self, MOF, ppm, replicate_id, time_adjust=0):
            self.MOF = MOF
            self.ppm = ppm
            self.replicate_id = replicate_id
            self.time_adjust = time_adjust

            try:
                self.data = read_data(MOF, ppm, time_adjust=self.time_adjust)[replicate_id]
            except IndexError:
                print(f"Error: replicate_id {replicate_id} does not exist for {MOF} at {ppm} ppm.")

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

            all_info = "{}_{}ppm_{}".format(self.MOF, self.ppm, self.replicate_id)
            plt.title(all_info)

            if save:
                plt.savefig("responses/featurized_{}.png".format(all_info), format="png")
            plt.show()
    return (SensorResponse,)


@app.cell
def _(SensorResponse):
    # Test the SensorResponse class initial_slope function
    _sensor_response = SensorResponse("Ni-HHTP", "0+40", 1)
    _sensor_response.compute_features()
    _sensor_response.viz(save=True)
    return


@app.cell
def _():
    # Read data from existing data in csv or loop through raw data?
    read_data_from_file = False
    return (read_data_from_file,)


@app.cell
def _():
    MOFs = ["Cu-HHTP", "Ni-HHTP", "Zn-HHTP"]
    features = ['auc', 'slope', 'saturation']
    ppms = ["0+5", "0+10", "0+20", "0+40", "5+0", "10+0", "20+0", "40+0", "4+36", "5+5", "9+13", 
            "13+27", "18+4", "22+18", "27+31", "31+9","36+22", "40+40"]
    return MOFs, features, ppms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Loop through raw data to compute all sensor responses""")
    return


@app.cell
def _(MOFs, SensorResponse, ppms, read_data_from_file):
    # list for data, will append cof, gas, carrier, and features of each sensor_response
    data = []
    for MOF in MOFs:
        for ppm in ppms:
            for rep_id in range(8):
                if read_data_from_file:
                    continue
                try:
                    sensor_response = SensorResponse(MOF, ppm, rep_id)
                    sensor_response.compute_features()
                    sensor_response.viz(save=True)
                    data.append([MOF, ppm, rep_id, sensor_response.slope_info['slope'],
                                sensor_response.saturation, sensor_response.auc]) 

                except (AttributeError, Exception):
                    pass
    return MOF, data, ppm, rep_id, sensor_response


@app.cell
def _(data, pd, read_data_from_file):
    # Put list of data into dataframe
    if read_data_from_file:
        data_df = pd.read_csv("responses.csv")
        data_df.drop(columns=['Unnamed: 0'], inplace=True) # remove index column, artifact of reading in
    else:
        data_df = pd.DataFrame(data, columns=['MOF', 'ppm', 'rep_id', 'slope', 'saturation', 'auc'])
        data_df.to_csv("responses.csv")
    data_df
    return (data_df,)


@app.cell
def _(data_df):
    data_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Time adjustment for sensor delay or human error""")
    return


@app.cell
def _(SensorResponse):
    # input data, experiment, and slope partition adjustment, output: dataframe and viz with adjusted slope feature
    def make_adjustment(data_df, MOF, ppm, rep_ids, n_partitions_slope_adj=15, n_partitions_saturation_adj=100, time_adjust=0):
        for rep_id in rep_ids:
            try:
                sensor_response = SensorResponse(MOF, ppm, rep_id, time_adjust=time_adjust)
                sensor_response.compute_features(n_partitions_slope=n_partitions_slope_adj,
                                                 n_partitions_saturation=n_partitions_saturation_adj)
                sensor_response.viz(save=True)
                data_df.loc[(data_df['MOF']==MOF)
                                    & (data_df['ppm']==ppm)
                                    & (data_df['rep_id']==rep_id), 'slope'] = sensor_response.slope_info['slope']
                data_df.loc[(data_df['MOF']==MOF)
                                    & (data_df['ppm']==ppm)
                                    & (data_df['rep_id']==rep_id), 'auc'] = sensor_response.auc
                data_df.loc[(data_df['MOF']==MOF)
                                    & (data_df['ppm']==ppm)
                                    & (data_df['rep_id']==rep_id), 'saturation'] = sensor_response.saturation
            except:
                pass
        return data_df
    return (make_adjustment,)


@app.cell
def _(data_df, make_adjustment):
    make_adjustment(data_df, MOF='Zn-HHTP', ppm="9+13", rep_ids=[0, 3], n_partitions_slope_adj=3)
    make_adjustment(data_df, MOF='Zn-HHTP', ppm="13+27", rep_ids=[1], n_partitions_slope_adj=3)
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="0+5", rep_ids=[2], n_partitions_slope_adj=3)
    make_adjustment(data_df, MOF='Ni-HHTP', ppm="5+0", rep_ids=[1], n_partitions_slope_adj=3)
    return


@app.cell
def _(data_df, make_adjustment):
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="9+13", rep_ids=[0, 1, 2, 3, 4], time_adjust=100)
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="13+27", rep_ids=[0, 1, 2, 3, 4], time_adjust=100)
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="18+4", rep_ids=[0, 1, 2, 3, 4], time_adjust=50)
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="27+31", rep_ids=[0, 1, 2, 3, 4], time_adjust=50)
    make_adjustment(data_df, MOF='Zn-HHTP', ppm="5+0", rep_ids=[0, 1, 2, 3], n_partitions_slope_adj=3)
    make_adjustment(data_df, MOF='Zn-HHTP', ppm="10+0", rep_ids=[0, 1, 2, 3], n_partitions_slope_adj=3)
    make_adjustment(data_df, MOF='Zn-HHTP', ppm="0+5", rep_ids=[0, 1, 2, 3], time_adjust=150)
    return


@app.cell
def _(data_df, make_adjustment):
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="0+10", rep_ids=[3], time_adjust=200)
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="0+5", rep_ids=[3], n_partitions_slope_adj=5)
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="10+0", rep_ids=[0, 1, 2], time_adjust=200)
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="20+0", rep_ids=[0, 1, 2, 3], time_adjust=200)
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="22+18", rep_ids=[0, 1, 2, 3, 4], time_adjust=100)
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="31+9", rep_ids=[0, 1, 2, 3, 4],  time_adjust=100)
    make_adjustment(data_df, MOF='Cu-HHTP', ppm="36+22", rep_ids=[0, 1, 2, 3, 4],  time_adjust=50)
    make_adjustment(data_df, MOF='Ni-HHTP', ppm="5+5", rep_ids=[0, 1, 2, 3],  time_adjust=100)
    make_adjustment(data_df, MOF='Ni-HHTP', ppm="31+9", rep_ids=[0, 1, 2, 3, 4], time_adjust=50)
    return


@app.cell
def _(data_df):
    # update csv with adjusted responses
    data_df.to_csv("responses.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Assemble and standardize complete array response vectors""")
    return


@app.cell
def _(MOFs, features, np, pd, ppms):
    def assemble_array_response(data_df, ppms=ppms, MOFs=MOFs, n_replicates=7, features=features):
        #  matrix will store response features.
        #  col = sensor array response vector
        #  row = particular response feature for a particular MOF (9. 3 MOFs x 3 feature each)
        #  loop through data to build matrix column by column (technically row by row and then transpose)

        matrix = []
        experiments = [] # List which will store experiment setup for each array column

        for ppm in ppms:
            for rep in range(n_replicates):
                col = []
                experiment = {'ppm': ppm,
                            'rep_id': rep}
                for MOF in MOFs:
                    for (i, feature) in enumerate(features):
                        try:
                            val = data_df.loc[(data_df['MOF']==MOF)
                                            & (data_df['ppm']==ppm)
                                            & (data_df['rep_id']==rep)][feature]
                            assert len(val) <= 1, "more than one instance"
                            col.append(val.iloc[0])
                        except (IndexError, KeyError):
                            pass

                    # only append column if entire array response exists
                    if len(col) == len(MOFs) * len(features):
                        matrix.append(col)
                        experiments.append(experiment)
                    else:
                        print("No complete array for experiment: ", experiment)

        matrix = np.array(matrix)
        response_array = pd.DataFrame(matrix)
        return experiments, response_array
    return (assemble_array_response,)


@app.cell
def _(PowerTransformer, assemble_array_response, data_df, pd):
    experiments, response_array = assemble_array_response(data_df)
    response_array = pd.DataFrame(PowerTransformer().fit_transform(response_array))
    return experiments, response_array


@app.cell
def _(experiments, pd):
    experiments_df = pd.DataFrame(experiments)
    return (experiments_df,)


@app.cell
def _(experiments_df, response_array):
    combo_df = experiments_df.join(response_array)
    return (combo_df,)


@app.cell
def _(combo_df):
    combo_df["H2S"] = combo_df["ppm"].apply(lambda x : int(x.split("+")[0]))
    combo_df["SO2"] = combo_df["ppm"].apply(lambda x : int(x.split("+")[1]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# create heatmap of sensor feature values""")
    return


@app.cell
def _(combo_df):
    # transpose to get complete arrays as columns for heatmap
    heatmatrixdf = combo_df.sort_values(by=["H2S", "SO2"], ascending=False) # sort by H2S and SO2 concentrations
    return (heatmatrixdf,)


@app.cell
def _():
    def gas_to_subscript(gas):
         sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
         return gas.translate(sub)
    return (gas_to_subscript,)


@app.cell
def _(
    LinearSegmentedColormap,
    features,
    gas_to_subscript,
    make_axes_locatable,
    np,
    plt,
    response_array,
    sns,
):
    def plot_heatmap(heatmatrixdf):
        RdGn = cmap = LinearSegmentedColormap.from_list("mycmap", ["red", "white", "green"])
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(28, 10), gridspec_kw={'height_ratios':[3, 1, 1]})

        # font size
        fs = 20
        yticklabels = features * 3

        # create heatmap
        heat_matrix_plot = heatmatrixdf[response_array.columns].T
        heat = sns.heatmap(heat_matrix_plot, cmap=RdGn, center=0, yticklabels=yticklabels, vmin=-2, vmax=2,
                         square=True, ax=ax1, cbar=False)

        # create a new axes for the colorbar
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2%", pad=0.8)  # increase pad to move colorbar further right

        # add colorbar to the new axes
        cbar = fig.colorbar(heat.collections[0], cax=cax)

        # adjust colorbar ticks
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_ticks([-2, -1, 0, 1, 2])

        # add colorbar label
        cbar.set_label(label='transformed response', size=fs)

        # label the MOFs:
        ax1.annotate('Cu', xy=(1.01, 7.5/9), xycoords='axes fraction',
                    fontsize=fs, ha='left', va='center',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=']- ,widthA=1.6, lengthA=1, angleA=180', lw=2, color='k'))

        ax1.annotate('Ni', xy=(1.01, 4.5/9), xycoords='axes fraction',
                    fontsize=fs, ha='left', va='center',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=']- ,widthA=1.6, lengthA=1, angleA=180', lw=2, color='k'))

        ax1.annotate('Zn', xy=(1.01, 1.5/9), xycoords='axes fraction',
                    fontsize=fs, ha='left', va='center',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=']- ,widthA=1.6, lengthA=1, angleA=180', lw=2, color='k'))

        ax1.set_xticks([])
        ax1.set_yticks(ax1.get_yticks())
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=fs)

        # create scatter ppm plot
        for (ax, gas) in zip((ax2, ax3), ("H2S", "SO2")):
            ax.scatter(x=np.arange(0, len(heatmatrixdf)), y=heatmatrixdf[gas], s=180, clip_on=False)
            ax.set_xlim(ax1.get_xlim())
            ax.set_ylabel(f"{gas_to_subscript(gas)} [ppm]", fontsize=fs)
            ax.tick_params(axis='both', which='both', labelsize=fs)
            ax.set_xticks(ticks=np.arange(0, len(heatmatrixdf)), labels=[])

            # adjust the position of ax2 to align with ax1
            pos1 = ax1.get_position()
            pos2 = ax.get_position()
            ax.set_position([pos1.x0-0.015, pos2.y0, pos1.width, pos2.height])


            # make ppm plot nice
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlim(left=-0.22)
            ax.set_ylim(top=40, bottom=-0.1)

            ax.grid(axis='x', color='grey')
            ax.set_yticks(ticks=[40, 20, 0])
        plt.savefig("heatmap.pdf", bbox_inches='tight', pad_inches=0.5)
        return plt.show()
    return (plot_heatmap,)


@app.cell
def _(heatmatrixdf, plot_heatmap):
    plot_heatmap(heatmatrixdf)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Experiment design space""")
    return


@app.cell
def _(np):
    def conc_to_class(combo_df):
        valid_region = (combo_df["H2S"] != 20.0) & (combo_df["SO2"] != 20.0)
        new_combo_df = combo_df[valid_region].copy()
        new_combo_df.reset_index(inplace=True)
        new_combo_df["target"] = str(np.zeros(sum(valid_region)))
        for i in range(len(new_combo_df)):
            if new_combo_df.loc[i, "H2S"] < 20 and new_combo_df.loc[i, "SO2"] < 20: 
                new_combo_df.loc[i, "target"] = "--"
            elif new_combo_df.loc[i, "H2S"] > 20 and new_combo_df.loc[i, "SO2"] < 20:
                new_combo_df.loc[i, "target"] = "+-"
            elif new_combo_df.loc[i, "H2S"] < 20 and new_combo_df.loc[i, "SO2"] > 20:
                new_combo_df.loc[i, "target"] = "-+"
            else:  
                new_combo_df.loc[i, "target"] = "++"
        return new_combo_df
    return (conc_to_class,)


@app.cell
def _(combo_df, conc_to_class):
    new_combo_df = conc_to_class(combo_df)
    return (new_combo_df,)


@app.cell
def _(combo_df):
    UQ_region = (combo_df["H2S"] == 20.0) | (combo_df["SO2"] == 20.0)
    return (UQ_region,)


@app.cell
def _(Line2D, UQ_region, combo_df, new_combo_df, plt, target_to_color):
    facecolors = [target_to_color[target] for target in new_combo_df["target"]]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(new_combo_df["H2S"], new_combo_df["SO2"], 
               clip_on=False, marker="o", s=100, facecolors=facecolors, edgecolors='black')
    ax.scatter(combo_df.loc[UQ_region, "H2S"], combo_df.loc[UQ_region, "SO2"], 
               clip_on=False, marker="s", s=100, facecolors='none', edgecolors='black')

    # legends
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', label='LOOCV',
               markerfacecolor='none', markeredgecolor='black', markersize=10, linestyle=''),
        Line2D([0], [0], marker='s', color='black', label='UQ test',
               markerfacecolor='none', markeredgecolor='black', markersize=10, linestyle='')
    ]

    # region plots
    alpha = 0.05
    ax.fill_between([0, 20], 0, y2=20, color=(target_to_color["--"], alpha))
    ax.fill_between([0, 20], 20, y2=40, color=(target_to_color["-+"], alpha))
    ax.fill_between([20, 40], 0, y2=20, color=(target_to_color["+-"], alpha))
    ax.fill_between([20, 40], 20, y2=40, color=(target_to_color["++"], alpha))

    ax.set_xlabel("H$_2$S [ppm]")
    ax.set_ylabel("SO$_2$ [ppm]")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_aspect('equal', "box")
    ax.legend(handles=legend_elements)
    plt.savefig("experiment_space.pdf")
    plt.show();
    return alpha, ax, facecolors, fig, legend_elements


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Principal Component Analysis (PCA)""")
    return


@app.cell
def _(PCA, new_combo_df, pd, response_array):
    pcadata = new_combo_df[response_array.columns].copy()

    pca = PCA(n_components=2)
    latent_vectors = pca.fit_transform(pcadata)
    z1, z2 = pca.explained_variance_ratio_
    print(z1, z2)

    pcs = pd.DataFrame(data=latent_vectors, columns = ['PC1', 'PC2'])
    pcs_and_exps = pd.concat([new_combo_df, pcs], axis = 1) # add principal components to f
    return latent_vectors, pca, pcadata, pcs, pcs_and_exps, z1, z2


@app.cell
def _(Line2D, plt, target_to_color, target_to_shape):
    def plot_PCA(pcs_and_exps, z1, z2, savename="PCA.pdf"):
        pc1 = pcs_and_exps['PC1']
        pc2 = pcs_and_exps['PC2']

        mixture_types = pcs_and_exps['target']
        unique_mixture = (("--", "H$_2$S$^-$, SO$_2$$^-$"), ("+-", "H$_2$S$^+$, SO$_2$$^-$"), 
                              ("-+", "H$_2$S$^-$, SO$_2$$^+$"), ("++", "H$_2$S$^+$, SO$_2$$^+$"))
        
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='grey', zorder=0)
        ax.axvline(x=0, color='grey', zorder=0)

        # create the bubble plot and legend handles
        mixture_legend_elements = []
        for mixture_type, label in unique_mixture:
            mixture_mask = (mixture_types == mixture_type)    
            scatter = ax.scatter(pc1[mixture_mask], pc2[mixture_mask], s=80,
                                edgecolors=target_to_color[mixture_type], marker=target_to_shape[mixture_type], 
                                 linewidths=1.5, facecolors='none')
            mixture_legend_elements.append(Line2D([0], [0], marker=target_to_shape[mixture_type], color='w', label=label,
                                            markeredgecolor=target_to_color[mixture_type], markerfacecolor='none', markersize=10))
            
        # set x and y axis labels and limits
        ax.set_xlabel(f'PC1 score, z$_1$ [{round(z1*100, 1)}%]')
        ax.set_ylabel(f'PC2 score, z$_2$ [{round(z2*100, 1)}%]')
        # ax.tick_params(axis="both", labelsize=15)
        ax.grid(False)

        # create the legends
        mixture_legend = ax.legend(handles=mixture_legend_elements, title=None, loc=(1.04,.5), frameon=False)

        ax.add_artist(mixture_legend)
        plt.axis('scaled')

        plt.tight_layout()

        # Adjust the layout
        plt.savefig(savename, bbox_extra_artists=(mixture_legend, ), bbox_inches='tight')

        return plt.show()
    return (plot_PCA,)


@app.cell
def _(pcs_and_exps, plot_PCA, z1, z2):
    plot_PCA(pcs_and_exps, z1, z2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Mixture classification (Random Forest)""")
    return


@app.cell
def _(RandomForestClassifier, new_combo_df, np, response_array):
    def classification():
        model_metric = {"true" : [], "pred" : []}
        features_importance = np.zeros(response_array.shape[1])
        for (H2S_ppm, SO2_ppm) in np.unique(new_combo_df[['H2S', 'SO2']].values, axis=0):
            test_ids = (new_combo_df["H2S"] == H2S_ppm) & (new_combo_df["SO2"] == SO2_ppm)
            train_ids = ~test_ids
            X_train = new_combo_df.loc[train_ids, response_array.columns]
            X_test = new_combo_df.loc[test_ids, response_array.columns]
            y_train = new_combo_df.loc[train_ids, "target"]
            y_test = new_combo_df.loc[test_ids, "target"]

            model_metric["true"].extend(y_test)
            clf = RandomForestClassifier(n_estimators=500, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            model_metric["pred"].extend(y_pred)
            features_importance += clf.feature_importances_
        return model_metric, features_importance / len(np.unique(new_combo_df[['H2S', 'SO2']].values, axis=0))
    return (classification,)


@app.cell
def _(classification):
    metric, features_importance = classification()
    return features_importance, metric


@app.cell
def _(confusion_matrix, metric):
    cm = confusion_matrix(metric["true"], metric["pred"])
    return (cm,)


@app.cell
def _(cm, plt, sns):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["H$_2$S$^+$, SO$_2$$^+$", "H$_2$S$^+$, SO$_2$$^-$", "H$_2$S$^-$, SO$_2$$^+$", "H$_2$S$^-$, SO$_2$$^-$"],
                yticklabels=["H$_2$S$^+$, SO$_2$$^+$", "H$_2$S$^+$, SO$_2$$^-$", "H$_2$S$^-$, SO$_2$$^+$", "H$_2$S$^-$, SO$_2$$^-$"],
                cbar_kws={"label":"# experiments"})


    plt.yticks(rotation=0)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.title('mixture class classification')
    plt.tight_layout()
    plt.savefig("cm.pdf")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Sensor importance""")
    return


@app.cell
def _(MOFs, features_importance, np, pd, plt, sns):
    def plot_MOF_importance(features_importance=features_importance, MOFs=MOFs):
        MOF_importance = np.zeros(len(MOFs))
        jump = len(features_importance) // len(MOFs) # features are order based on how we iterate over MOFs.
        for (j, MOF) in enumerate(MOFs): 
            MOF_importance[j] = sum(features_importance[j * jump : j * jump + jump])
        MOF_importance = pd.DataFrame({"sensor importance score" : MOF_importance, 
                                       "MOF" : ["Zn$_3$(HHTP)$_2$", "Cu$_3$(HHTP)$_2$", "Ni$_3$(HHTP)$_2$"]})
        MOF_importance.sort_values(by="sensor importance score", inplace=True, ascending=False)

        fig, ax = plt.subplots(1, 1)
        sns.barplot(MOF_importance, x="MOF", y="sensor importance score", edgecolor="black", facecolor="white")
        ax.set_ylim(0, 1)
        plt.savefig("sensor_importance.pdf")
        return plt.show()
    return (plot_MOF_importance,)


@app.cell
def _(plot_MOF_importance):
    plot_MOF_importance()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# model uncertainity quantification""")
    return


@app.cell
def _(RandomForestClassifier, new_combo_df, response_array):
    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf.fit(new_combo_df[response_array.columns], new_combo_df["target"])
    return (clf,)


@app.cell
def _(UQ_region, clf, combo_df, np, response_array):
    UQ_pred = []
    for estimator in clf.estimators_:
        pred = estimator.predict(combo_df.loc[UQ_region, response_array.columns])
        UQ_pred.append(pred)
    UQ_pred = np.vstack(UQ_pred)
    return UQ_pred, estimator, pred


@app.cell
def _(UQ_pred, UQ_region, np):
    class_dist = np.zeros((sum(UQ_region), 4))
    for j in range(UQ_pred.shape[1]):
        counts = np.zeros(4)
        classes, count = np.unique(UQ_pred[:, j], return_counts=True)
        counts[classes.astype(int)] = count
        class_dist[j] = counts
    class_dist /= 5 # covert to percentage
    return class_dist, classes, count, counts, j


@app.cell
def _(UQ_region, combo_df):
    UQ_classification = combo_df.loc[UQ_region, ["H2S", "SO2"]]
    return (UQ_classification,)


@app.cell
def _(UQ_classification, class_dist, clf):
    UQ_classification[clf.classes_] = class_dist
    UQ_classification.reset_index(drop=True, inplace=True)
    return


@app.cell
def _(UQ_classification):
    UQ_classification
    return


if __name__ == "__main__":
    app.run()
