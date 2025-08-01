import marimo

__generated_with = "0.13.13"
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
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.decomposition import PCA

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    return (
        Line2D,
        LinearSegmentedColormap,
        PCA,
        Path,
        PowerTransformer,
        RandomForestClassifier,
        RandomForestRegressor,
        confusion_matrix,
        make_axes_locatable,
        np,
        pd,
        plt,
        r2_score,
        sns,
    )


@app.cell
def _():
    from utils import _find_ppm_sheet, linear_regression, SensorResponse, _find_header_row
    return (SensorResponse,)


@app.cell
def _(plt):
    plt.rcParams.update({"font.size": 18})
    return


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
        read_data(MOF, ppm, time_adjust)

    read in the sensor response data for a given MOF exposed to a
    given gas mixture concentration.
    returns list of pandas data frames with this data. (may be replicates)
    each data frame has two columns: time, DeltaG/G0.
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
                ppm_sheet = _find_ppm_sheet(filename)
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


@app.cell
def _(SensorResponse, read_data):
    # Test the SensorResponse class initial_slope function
    _data = read_data("Ni-HHTP", "0+40")[1]
    _title = "{}_{}ppm_{}".format("Ni-HHTP", "0+40", 1)
    _sensor_response = SensorResponse(_data, _title)
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
def _(MOFs, SensorResponse, ppms, read_data, read_data_from_file):
    # list for data, will append MOF, ppm, and features of each sensor_response
    raw_data = []
    if not read_data_from_file:
        for MOF in MOFs:
            for ppm in ppms:
                for rep_id in range(8):
                    try:
                        this_data = read_data(MOF, ppm)[rep_id]
                        this_title = "{}_{}ppm_{}".format(MOF, ppm, rep_id)
                        sensor_response = SensorResponse(this_data, this_title)
                        sensor_response.compute_features()
                        sensor_response.viz(save=True)
                        raw_data.append([MOF, ppm, rep_id, sensor_response.slope_info['slope'],
                                    sensor_response.saturation, sensor_response.auc]) 
                    except (AttributeError, Exception):
                        pass
    return (raw_data,)


@app.cell
def _(pd, raw_data, read_data_from_file):
    # Put list of data into dataframe
    if not read_data_from_file:
        prelim_data = pd.DataFrame(raw_data, columns=['MOF', 'ppm', 'rep_id', 'slope', 'saturation', 'auc'])

        prelim_data # b/c we'll make adjustements later.
    return (prelim_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Time adjustment for sensor delay or human error""")
    return


@app.cell
def _(SensorResponse, read_data):
    # input data, experiment, and slope partition adjustment, output: dataframe and viz with adjusted slope feature
    def make_adjustment(
        prelim_data, MOF, ppm, rep_ids, 
        n_partitions_slope_adj=15, n_partitions_saturation_adj=100, time_adjust=0
    ):
        for rep_id in rep_ids:
            try:
                this_data = read_data(MOF, ppm, time_adjust=time_adjust)[rep_id]
                this_title = "{}_{}ppm_{}".format(MOF, ppm, rep_id)
                sensor_response = SensorResponse(this_data, this_title)
                sensor_response.compute_features(n_partitions_slope=n_partitions_slope_adj,
                                                 n_partitions_saturation=n_partitions_saturation_adj)
                sensor_response.viz(save=True)
                prelim_data.loc[(prelim_data['MOF']==MOF)
                                    & (prelim_data['ppm']==ppm)
                                    & (prelim_data['rep_id']==rep_id), 'slope'] = sensor_response.slope_info['slope']
                prelim_data.loc[(prelim_data['MOF']==MOF)
                                    & (prelim_data['ppm']==ppm)
                                    & (prelim_data['rep_id']==rep_id), 'auc'] = sensor_response.auc
                prelim_data.loc[(prelim_data['MOF']==MOF)
                                    & (prelim_data['ppm']==ppm)
                                    & (prelim_data['rep_id']==rep_id), 'saturation'] = sensor_response.saturation
            except:
                pass
        return prelim_data
    return (make_adjustment,)


@app.cell
def _(make_adjustment, pd, prelim_data, read_data_from_file):
    # do all of these in one cell.
    if not read_data_from_file:
        data = prelim_data.copy()
        make_adjustment(data, MOF='Zn-HHTP', ppm="9+13", rep_ids=[0, 3], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='Zn-HHTP', ppm="13+27", rep_ids=[1], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='Cu-HHTP', ppm="0+5", rep_ids=[2], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='Ni-HHTP', ppm="5+0", rep_ids=[1], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='Cu-HHTP', ppm="9+13", rep_ids=[0, 1, 2, 3, 4], time_adjust=100)
        make_adjustment(data, MOF='Cu-HHTP', ppm="13+27", rep_ids=[0, 1, 2, 3, 4], time_adjust=100)
        make_adjustment(data, MOF='Cu-HHTP', ppm="18+4", rep_ids=[0, 1, 2, 3, 4], time_adjust=50)
        make_adjustment(data, MOF='Cu-HHTP', ppm="27+31", rep_ids=[0, 1, 2, 3, 4], time_adjust=50)
        make_adjustment(data, MOF='Zn-HHTP', ppm="5+0", rep_ids=[0, 1, 2, 3], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='Zn-HHTP', ppm="10+0", rep_ids=[0, 1, 2, 3], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='Zn-HHTP', ppm="0+5", rep_ids=[0, 1, 2, 3], time_adjust=150)
        make_adjustment(data, MOF='Cu-HHTP', ppm="0+10", rep_ids=[3], time_adjust=200)
        make_adjustment(data, MOF='Cu-HHTP', ppm="0+5", rep_ids=[3], n_partitions_slope_adj=5)
        make_adjustment(data, MOF='Cu-HHTP', ppm="10+0", rep_ids=[0, 1, 2], time_adjust=200)
        make_adjustment(data, MOF='Cu-HHTP', ppm="20+0", rep_ids=[0, 1, 2, 3], time_adjust=200)
        make_adjustment(data, MOF='Cu-HHTP', ppm="22+18", rep_ids=[0, 1, 2, 3, 4], time_adjust=100)
        make_adjustment(data, MOF='Cu-HHTP', ppm="31+9", rep_ids=[0, 1, 2, 3, 4],  time_adjust=100)
        make_adjustment(data, MOF='Cu-HHTP', ppm="36+22", rep_ids=[0, 1, 2, 3, 4],  time_adjust=50)
        make_adjustment(data, MOF='Ni-HHTP', ppm="5+5", rep_ids=[0, 1, 2, 3],  time_adjust=100)
        make_adjustment(data, MOF='Ni-HHTP', ppm="31+9", rep_ids=[0, 1, 2, 3, 4], time_adjust=50)

        # save
        data.to_csv("mixture_responses.csv")
    else:
        data = pd.read_csv("mixture_responses.csv") # this is adjusted.
        data.drop(columns=['Unnamed: 0'], inplace=True) # remove index column, artifact of reading in
    return (data,)


@app.cell
def _(data):
    data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Assemble and standardize complete array response vectors""")
    return


@app.cell
def _(MOFs, features):
    feature_col_names = [MOF + " " + feature for MOF in MOFs for feature in features]
    return (feature_col_names,)


@app.cell
def _(MOFs, feature_col_names, features, np, pd, ppms):
    def assemble_array_response(data, ppms=ppms, MOFs=MOFs, n_replicates=7, features=features):
        #  matrix will store response features.
        #  col = sensor array response vector
        #  row = particular response feature for a particular MOF (9. 3 MOFs x 3 feature each)
        #  loop through data to build matrix column by column (technically row by row and then transpose)

        matrix = []
        experiments = [] # List which will store experiment setup for each array column

        for ppm in ppms:
            for rep in range(n_replicates):
                col = []
                experiment = {'ppm': ppm, 'rep_id': rep}
                for MOF in MOFs:
                    for (i, feature) in enumerate(features):
                        try:
                            val = data.loc[(data['MOF']==MOF)
                                            & (data['ppm']==ppm)
                                            & (data['rep_id']==rep)][feature]
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

        # join experiments and responses in one combo data frame.
        matrix = np.array(matrix)
        response_array = pd.DataFrame(matrix, columns=feature_col_names)
        combo_df = pd.DataFrame(experiments).join(response_array)

        # H2S, SO2 columns
        combo_df["H2S"] = combo_df["ppm"].apply(lambda x : int(x.split("+")[0]))
        combo_df["SO2"] = combo_df["ppm"].apply(lambda x : int(x.split("+")[1]))

        return combo_df
    return (assemble_array_response,)


@app.cell
def _(assemble_array_response, data):
    combo_df = assemble_array_response(data)
    combo_df
    return (combo_df,)


@app.cell
def _(PowerTransformer, combo_df, feature_col_names):
    transformed_combo_df = combo_df.copy()
    transformed_combo_df[feature_col_names] = PowerTransformer().fit_transform(transformed_combo_df[feature_col_names])
    transformed_combo_df
    return (transformed_combo_df,)


@app.cell
def _():
    # feature correlation plot
    return


@app.cell
def _(MOFs, feature_col_names, features, pd, transformed_combo_df):
    all_features_df = pd.DataFrame()
    for i in range(len(MOFs)):
        start = i * len(features)
        end = start + len(features)
    
        sub_col = feature_col_names[start:end]
        MOF_df = transformed_combo_df[sub_col]
        MOF_df.columns = ["AUC", "slope", "saturation"]
    
        all_features_df = pd.concat([all_features_df, MOF_df])
    return (all_features_df,)


@app.cell
def _(all_features_df, plt, sns):
    sns.pairplot(all_features_df)
    plt.savefig("feature_pairplot.pdf")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Experiment design space""")
    return


@app.cell
def _(np):
    def get_classification_data(combo_df):
        # cut out UQ points
        loo_data = (combo_df["H2S"] != 20.0) & (combo_df["SO2"] != 20.0)

        new_combo_df = combo_df[loo_data].copy()
        new_combo_df.reset_index(inplace=True)
        new_combo_df["target"] = str(np.zeros(sum(loo_data)))
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
    return (get_classification_data,)


@app.cell
def _(get_classification_data, transformed_combo_df):
    pca_combo_df = get_classification_data(transformed_combo_df)
    return (pca_combo_df,)


@app.cell
def _(combo_df):
    UQ_region = (combo_df["H2S"] == 20.0) | (combo_df["SO2"] == 20.0)
    return (UQ_region,)


@app.cell
def _():
    rf_class_to_pretty_name = {
        "++": "SO$_2\u26A0$ & H$_2$S$\u26A0$",
        "-+": "SO$_2\u26A0$ & H$_2$S$\u2713$",
        "--": "SO$_2\u2713$ & H$_2$S$\u2713$",
        "+-": "SO$_2\u2713$ & H$_2$S$\u26A0$",
    }
    rf_class_to_pretty_name
    return (rf_class_to_pretty_name,)


@app.cell
def _(
    Line2D,
    UQ_region,
    combo_df,
    pca_combo_df,
    plt,
    rf_class_to_pretty_name,
    target_to_color,
):
    facecolors = [target_to_color[target] for target in pca_combo_df["target"]]
    fs = 14

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(pca_combo_df["H2S"], pca_combo_df["SO2"], 
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
    alpha = 0.1
    ax.fill_between([0, 20], 0, y2=20, color=(target_to_color["--"], alpha))
    ax.text(10, 16, rf_class_to_pretty_name["--"], fontsize=fs, horizontalalignment="center", verticalalignment="center")
    ax.fill_between([0, 20], 20, y2=40, color=(target_to_color["-+"], alpha))
    ax.text(10, 24, rf_class_to_pretty_name["-+"], fontsize=fs, horizontalalignment="center", verticalalignment="center")
    ax.fill_between([20, 40], 0, y2=20, color=(target_to_color["+-"], alpha))
    ax.text(30, 16, rf_class_to_pretty_name["+-"], fontsize=fs, horizontalalignment="center", verticalalignment="center")
    ax.fill_between([20, 40], 20, y2=40, color=(target_to_color["++"], alpha))
    ax.text(30, 24, rf_class_to_pretty_name["++"], fontsize=fs, horizontalalignment="center", verticalalignment="center")

    plt.plot([20, 20], [0, 40], color="black", lw=1, linestyle="dashed")
    plt.plot([0, 40], [20, 20], color="black", lw=1, linestyle="dashed")

    ax.set_xlabel("H$_2$S [ppm]")
    ax.set_ylabel("SO$_2$ [ppm]")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_aspect('equal', "box")
    ax.legend(handles=legend_elements, fontsize=14, loc=(0.55, 0.82))
    plt.tight_layout()
    plt.savefig("experiment_space.pdf")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# create heatmap of sensor feature values""")
    return


@app.function
def gas_to_subscript(gas):
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return gas.translate(sub)


@app.cell
def _():
    gas_to_color = {"SO2" : "orange", "H2S" : "yellow"}
    return (gas_to_color,)


@app.cell
def _(
    LinearSegmentedColormap,
    feature_col_names,
    features,
    gas_to_color,
    make_axes_locatable,
    np,
    plt,
    sns,
):
    def plot_heatmap(pca_combo_df):
        heatmatrixdf = pca_combo_df.sort_values(by=["H2S", "SO2"], ascending=False)

        RdGn = cmap = LinearSegmentedColormap.from_list("mycmap", ["red", "white", "green"])
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(28, 10), gridspec_kw={'height_ratios':[3, 1, 1]})

        # font size
        fs = 30
        yticklabels = features * 3

        # create heatmap
        heat_matrix_plot = heatmatrixdf[feature_col_names].T
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
        cbar.set_label(label='transformed\nresponse', size=fs)

        # label the MOFs:
        ax1.annotate('Cu', xy=(1.01, 7.5/9), xycoords='axes fraction',
                    fontsize=fs, ha='left', va='center',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=']- ,widthA=1.15, lengthA=1, angleA=180', lw=2, color='k'))

        ax1.annotate('Ni', xy=(1.01, 4.5/9), xycoords='axes fraction',
                    fontsize=fs, ha='left', va='center',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=']- ,widthA=1.15, lengthA=1, angleA=180', lw=2, color='k'))

        ax1.annotate('Zn', xy=(1.01, 1.5/9), xycoords='axes fraction',
                    fontsize=fs, ha='left', va='center',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=']- ,widthA=1.15, lengthA=1, angleA=180', lw=2, color='k'))

        ax1.set_xticks([])
        ax1.set_yticks(ax1.get_yticks())
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=fs)

        # create scatter ppm plot
        for (ax, gas) in zip((ax2, ax3), ("SO2", "H2S")):
            ax.scatter(x=np.arange(0, len(heatmatrixdf)), y=heatmatrixdf[gas], s=180, color=gas_to_color[gas], 
                       edgecolor="black", clip_on=False, zorder=3)
            ax.set_xlim(ax1.get_xlim())
            ax.set_ylabel(f"{gas_to_subscript(gas)}\n[ppm]", fontsize=fs)
            ax.tick_params(axis='both', which='both', labelsize=fs)
            ax.set_xticks(ticks=np.arange(0, len(heatmatrixdf)), labels=[], zorder=0)

            # adjust the position of ax2 to align with ax1
            pos1 = ax1.get_position()
            pos2 = ax.get_position()
            ax.set_position([pos1.x0, pos2.y0, pos1.width - 0.035, pos2.height])


            # make ppm plot nice
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlim(left=-0.22)
            ax.set_ylim(top=50, bottom=-0.1)

            ax.grid(axis='x', color='grey')
            ax.set_yticks(ticks=[40, 20, 0])
        plt.savefig("heatmap.pdf", bbox_inches='tight', pad_inches=0.5)
        return plt.show()
    return (plot_heatmap,)


@app.cell
def _(pca_combo_df, plot_heatmap, plt):
    with plt.rc_context({'font.size': 38}):
        plot_heatmap(pca_combo_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Principal Component Analysis (PCA)""")
    return


@app.cell
def _(PCA, feature_col_names, pca_combo_df, pd):
    pcadata = pca_combo_df[feature_col_names].copy()

    pca = PCA(n_components=2)
    latent_vectors = pca.fit_transform(pcadata)
    z1, z2 = pca.explained_variance_ratio_
    print(z1, z2)

    pcs = pd.DataFrame(data=latent_vectors, columns=['PC1', 'PC2'])
    pcs_and_exps = pd.concat([pca_combo_df, pcs], axis=1)
    pcs_and_exps
    return pcs_and_exps, z1, z2


@app.cell
def _(Line2D, plt, rf_class_to_pretty_name, target_to_color, target_to_shape):
    def plot_PCA(pcs_and_exps, z1, z2, savename="PCA.pdf", rf_class_to_pretty_name=rf_class_to_pretty_name):
        pc1 = pcs_and_exps['PC1']
        pc2 = pcs_and_exps['PC2']

        mixture_types = pcs_and_exps['target']

        fig, ax = plt.subplots()
        ax.axhline(y=0, color='grey', zorder=0)
        ax.axvline(x=0, color='grey', zorder=0)

        # create the bubble plot and legend handles
        mixture_legend_elements = []
        for mixture_type in ["++", "+-", "-+", "--"]:
            label = rf_class_to_pretty_name[mixture_type]
            mixture_mask = (mixture_types == mixture_type)
            marker_size = pcs_and_exps.loc[mixture_mask, "H2S"] + pcs_and_exps.loc[mixture_mask, "SO2"]
            scatter = ax.scatter(pc1[mixture_mask], pc2[mixture_mask], s= 4 * marker_size,
                                 edgecolors=target_to_color[mixture_type], 
                                 marker=target_to_shape[mixture_type], linewidths=1.5, facecolors='none')
            mixture_legend_elements.append(Line2D([0], [0], marker=target_to_shape[mixture_type], color='w', label=label,
                                            markeredgecolor=target_to_color[mixture_type], markerfacecolor='none', markersize=10))

        # set x and y axis labels and limits
        ax.set_xlabel(f'PC1 score [{round(z1*100, 1)}%]')
        ax.set_ylabel(f'PC2 score [{round(z2*100, 1)}%]')
        ax.set_aspect('equal', "box")
        ax.grid(False)

        # create the legends
        mixture_legend = ax.legend(handles=mixture_legend_elements, title=None, loc=(.65,.01), frameon=True, fontsize=12)

        ax.add_artist(mixture_legend)
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
def _(combo_df, get_classification_data):
    rf_df = get_classification_data(combo_df) # un-normalized!
    return (rf_df,)


@app.cell
def _(rf_df):
    rf_df
    return


@app.cell
def _(RandomForestClassifier, clf, feature_col_names, np):
    def loo_classification(rf_df):
        parity_data = {"true" : [], "pred" : []}
        features_importance = np.zeros(len(feature_col_names))
        # loop over (H2S, SO2) concentrations
        for (SO2_ppm, H2S_ppm) in np.unique(rf_df[['SO2', 'H2S']].values, axis=0):
            # multiple replicates may be in test set.
            test_ids = (rf_df["H2S"] == H2S_ppm) & (rf_df["SO2"] == SO2_ppm)
            train_ids = ~test_ids

            X_train = rf_df.loc[train_ids, feature_col_names]
            X_test = rf_df.loc[test_ids, feature_col_names]

            model = RandomForestClassifier(n_estimators=500, random_state=0)
            y_train = rf_df.loc[train_ids, "target"]
            y_test = rf_df.loc[test_ids, "target"]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # track (y_true, y_pred) parity plot
            parity_data["true"].extend(y_test)
            parity_data["pred"].extend(y_pred)

            features_importance += model.feature_importances_

        return parity_data, features_importance / len(np.unique(rf_df[['SO2', 'H2S']].values, axis=0)), clf.classes_
    return (loo_classification,)


@app.cell
def _(loo_classification, rf_df):
    parity_data, features_importance, rf_classes = loo_classification(rf_df)
    return features_importance, parity_data, rf_classes


@app.cell
def _(confusion_matrix, parity_data, pd, rf_class_to_pretty_name, rf_classes):
    cm = pd.DataFrame(
        confusion_matrix(parity_data["true"], parity_data["pred"], labels=rf_classes), 
        columns=[rf_class_to_pretty_name[c] for c in rf_classes]
    )
    cm.index = [rf_class_to_pretty_name[c] for c in rf_classes]
    return (cm,)


@app.cell
def _(cm):
    cm
    return


@app.cell
def _(cm, plt, sns):
    with plt.rc_context({'font.size': 14}):
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Grays', 
            annot_kws ={"fontsize":19},
            cbar_kws={"label":"# experiments"},
            square=True
        )

        plt.yticks(rotation=0)
        plt.xlabel('predicted')
        plt.ylabel('true')

        plt.tight_layout()
        plt.savefig("cm.pdf")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Sensor importance""")
    return


@app.cell
def _():
    mof_to_pretty_name = {
        "Cu-HHTP": "Cu$_3$(HHTP)$_2$",
        "Ni-HHTP": "Ni$_3$(HHTP)$_2$",
        "Zn-HHTP": "Zn$_3$(HHTP)$_2$"
    }
    return (mof_to_pretty_name,)


@app.cell
def _(features, mof_to_pretty_name, np, pd, plt, sns):
    def plot_MOF_importance(features_importance, MOFs):
        MOF_importance = np.zeros(len(MOFs))

        jump = len(features_importance) // len(MOFs) # features are order based on how we iterate over MOFs.
        assert jump == len(features)

        for (j, MOF) in enumerate(MOFs): 
            MOF_importance[j] = sum(features_importance[j * jump : j * jump + jump])

        MOF_importance = pd.DataFrame(
            {"importance score" : MOF_importance, 
            "MOF" : [mof_to_pretty_name[mof] for mof in MOFs]}
        )
        MOF_importance.sort_values(by="importance score", inplace=True, ascending=False)

        fig, ax = plt.subplots(1, 1)
        sns.barplot(MOF_importance, x="MOF", y="importance score", edgecolor="black", facecolor="white")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig("sensor_importance.pdf")
        return plt.show()
    return (plot_MOF_importance,)


@app.cell
def _(MOFs, features_importance, plot_MOF_importance):
    plot_MOF_importance(features_importance, MOFs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# feature importance""")
    return


@app.cell
def _(features):
    features
    return


@app.cell
def _(MOFs, np, pd, plt, sns):
    def plot_feature_importance(features_importance, features, MOFs=MOFs):
        feature_importance = np.zeros(len(features))
        for (j, score) in enumerate(features_importance):
            idx = j % len(MOFs)
            feature_importance[idx] += score

        feature_importance = pd.DataFrame(
            {"importance score" : feature_importance, 
            "feature" : ["AUC", "slope", "saturation"]}
        )
        feature_importance.sort_values(by="importance score", inplace=True, ascending=False)

        fig, ax = plt.subplots(1, 1)
        sns.barplot(feature_importance, x="feature", y="importance score", edgecolor="black", facecolor="white")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig("feature_importance.pdf")
        return plt.show()
    return (plot_feature_importance,)


@app.cell
def _(features, features_importance, plot_feature_importance):
    plot_feature_importance(features_importance, features)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# prediction of mixture concentration""")
    return


@app.cell
def _():
    convex_hull = [[40, 0], [40, 40], [5, 0], [0, 5], [0, 40]] # to combat random forest extrapolation deficiency
    return (convex_hull,)


@app.cell
def _(RandomForestRegressor, convex_hull, feature_col_names, np):
    def loo_regression(combo_df, convex_hull=convex_hull):
        parity_data = {"true" : [], "pred" : []}
        # loop over (H2S, SO2) concentrations
        for (SO2_ppm, H2S_ppm) in np.unique(combo_df[['SO2', 'H2S']].values, axis=0):
            if [SO2_ppm, H2S_ppm] in convex_hull:
                continue
            # multiple replicates may be in test set.
            test_ids = (combo_df["H2S"] == H2S_ppm) & (combo_df["SO2"] == SO2_ppm)
            train_ids = ~test_ids

            X_train = combo_df.loc[train_ids, feature_col_names]
            X_test = combo_df.loc[test_ids, feature_col_names]

            model = RandomForestRegressor(n_estimators=500, random_state=0)
            y_train = combo_df.loc[train_ids, ['SO2', 'H2S']]
            y_test = combo_df.loc[test_ids, ['SO2', 'H2S']].values

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # track (y_true, y_pred) parity plot
            parity_data["true"].extend(y_test)
            parity_data["pred"].extend(y_pred)

        return parity_data
    return (loo_regression,)


@app.cell
def _(combo_df, loo_regression):
    reg_parity = loo_regression(combo_df)
    return (reg_parity,)


@app.cell
def _(np):
    def MAE(true, pred):
        diff = abs(true - pred)
        return np.mean(diff)
    return (MAE,)


@app.cell
def _(MAE, gas_to_color, np, plt, r2_score, sns):
    def viz_gas_concentration_prediction(reg_parity):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        for i, gas in enumerate(['SO2', 'H2S']):
            true, pred = np.vstack(reg_parity["true"])[:, i], np.vstack(reg_parity["pred"])[:, i]
            r2 = r2_score(true, pred)

            clip = max(max(true), max(pred))
            axs[i].plot([0.0, clip + 1], [0.0, clip + 1], linestyle="dashed", color="grey")
            sns.scatterplot(x=true, y=pred, s=100, ax=axs[i], zorder=1, clip_on=False, color=gas_to_color[gas], edgecolor="black")

            axs[i].set_xlim(0, clip + 1), 
            axs[i].set_ylim(0, clip + 1)
            axs[i].set_aspect('equal', "box")

            ticks = np.arange(0, round(clip + 1), 5)
            axs[i].set_yticks(ticks)
            axs[i].set_xticks(ticks)


            axs[i].set_ylabel("pred. concentration [ppm]")
            axs[i].set_xlabel("true concentration [ppm]")
            axs[i].set_title(gas_to_subscript(gas))


            textstr = "\n".join(
                (  
                    "MAE = %.2fppm" % MAE(true, pred),
                    r"R$^2=%.2f$" % r2,
                )
            )

            props = dict(boxstyle="round", facecolor="white", alpha=0.3)

            axs[i].text(
                0.04,
                0.95,
                textstr,
                transform=axs[i].transAxes,
                fontsize=12,
                verticalalignment="top",
                    bbox=props,
                )
        plt.savefig("gas_concentration_prediction.pdf")
        return fig
    return (viz_gas_concentration_prediction,)


@app.cell
def _(reg_parity, viz_gas_concentration_prediction):
    viz_gas_concentration_prediction(reg_parity)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# model uncertainity quantification""")
    return


@app.cell
def _(RandomForestClassifier, feature_col_names, rf_df):
    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf.fit(rf_df[feature_col_names], rf_df["target"])
    return (clf,)


@app.cell
def _(UQ_region, clf, combo_df, feature_col_names, np):
    UQ_pred = []
    for estimator in clf.estimators_:
        pred = estimator.predict(combo_df.loc[UQ_region, feature_col_names].values)
        UQ_pred.append(pred)
    UQ_pred = np.vstack(UQ_pred)
    UQ_pred
    return (UQ_pred,)


@app.cell
def _(UQ_pred, UQ_region, np):
    class_dist = np.zeros((sum(UQ_region), 4))
    for j in range(UQ_pred.shape[1]): # j is a data point in the UQ region.
        counts = np.zeros(4) # counts votes for each class, among the trees.
        # UQ_pred[:, j] is predictions of all of the trees for data point j.
        classes, count = np.unique(UQ_pred[:, j], return_counts=True)
        counts[classes.astype(int)] = count
        class_dist[j] = counts
    class_dist = class_dist / UQ_pred.shape[0] * 100 # covert to percentage
    return (class_dist,)


@app.cell
def _(UQ_region, combo_df):
    UQ_classification = combo_df.loc[UQ_region, ["H2S", "SO2"]]
    return (UQ_classification,)


@app.cell
def _(UQ_classification, class_dist, clf):
    UQ_classification[clf.classes_] = class_dist
    UQ_classification.reset_index(drop=True, inplace=True)
    UQ_classification
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    notes: 

    * for 20 ppm SO2, we'd expect some miss-classification as -+ and --. since def H2S is not there. but we do get some sizeable +- predictions...
    * for 20 ppm H2S, we'd expect some miss-classification as +- and --. since def SO2 is not there. this is largely indeed the case!
    """
    )
    return


if __name__ == "__main__":
    app.run()
