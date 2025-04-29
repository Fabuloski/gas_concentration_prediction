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
def _():
    from utils import _find_ppm_sheet, linear_regression, SensorResponse, _find_header_row
    return SensorResponse, linear_regression


@app.cell
def _(plt):
    my_colors = plt.get_cmap('tab10')
    my_colors
    return (my_colors,)


@app.cell
def _():
    gas_to_color = {"SO2" : "orange",
                  "H2S" : "yellow",
                  "NO" : "blue",
                  "CO" : "grey",
                  "NH3" : "purple"}
    return (gas_to_color,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Helper function to extract response data from excel file""")
    return


@app.cell
def _(Path, np, pd):
    """
        read_data(MOF, gas, ppm, time_adjust)

    read in the sensor response data for a given MOF exposed to a
    given gas mixture concentration.
    returns list of pandas data frames with this data. (may be replicates)
    each data frame has two columns: time, DeltaG/G0.
    """
    def read_data(MOF, gas, ppm, time_adjust=0):
        ppms = [5, 10, 20, 40, 80]

        path = Path.cwd().joinpath("pure_gases_data", gas, MOF, f"{ppm}ppm").rglob("*.xlsx")

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
    _data = read_data("Ni-HHTP", "CO", 40)[1]
    _title = "{}_{}_{}ppm_{}".format("Ni-HHTP", "CO", 40, 1)
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
    ppms = [5, 10, 20, 40, 80]
    gases = ["SO2", "H2S", "NO", "CO", "NH3"]
    return MOFs, features, gases, ppms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Loop through raw data to compute all sensor responses""")
    return


@app.cell
def _(MOFs, SensorResponse, gases, ppms, read_data, read_data_from_file):
    # list for data, will append MOF, gas, and features of each sensor_response
    raw_data = []
    for gas in gases:
        for MOF in MOFs:
            for ppm in ppms:
                for rep_id in range(8):
                    if read_data_from_file:
                        continue
                    try:
                        this_data = read_data(MOF, gas, ppm)[rep_id]
                        this_title = "{}_{}_{}ppm_{}".format(MOF, gas, ppm, rep_id)
                        sensor_response = SensorResponse(this_data, this_title)
                        sensor_response.compute_features()
                        sensor_response.viz(save=True)
                        raw_data.append([MOF, gas, ppm, rep_id, sensor_response.slope_info['slope'],
                                    sensor_response.saturation, sensor_response.auc]) 

                    except (AttributeError, Exception):
                        pass
    return (
        MOF,
        gas,
        ppm,
        raw_data,
        rep_id,
        sensor_response,
        this_data,
        this_title,
    )


@app.cell
def _(pd, raw_data, read_data_from_file):
    # Put list of data into dataframe
    if not read_data_from_file:
        prelim_data = pd.DataFrame(raw_data, columns=['MOF', 'gas', 'ppm', 'rep_id', 'slope', 'saturation', 'auc'])

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
        prelim_data, gas, MOF, ppm, rep_ids, 
        n_partitions_slope_adj=15, n_partitions_saturation_adj=100, time_adjust=0
    ):
        for rep_id in rep_ids:
            try:
                this_data = read_data(MOF, gas, ppm, time_adjust=time_adjust)[rep_id]
                this_title = "{}_{}_{}ppm_{}".format(MOF, gas, ppm, rep_id)
                sensor_response = SensorResponse(this_data, this_title)
                sensor_response.compute_features(n_partitions_slope=n_partitions_slope_adj,
                                                 n_partitions_saturation=n_partitions_saturation_adj)
                sensor_response.viz(save=True)
                prelim_data.loc[(prelim_data['MOF']==MOF)
                                    & (prelim_data['gas']==gas)
                                    & (prelim_data['ppm']==ppm)
                                    & (prelim_data['rep_id']==rep_id), 'slope'] = sensor_response.slope_info['slope']
                prelim_data.loc[(prelim_data['MOF']==MOF)
                                    & (prelim_data['ppm']==ppm)
                                    & (prelim_data['gas']==gas)
                                    & (prelim_data['rep_id']==rep_id), 'auc'] = sensor_response.auc
                prelim_data.loc[(prelim_data['MOF']==MOF)
                                    & (prelim_data['gas']==gas)
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
        make_adjustment(data, gas="SO2", MOF='Cu-HHTP', ppm=5, rep_ids=[2], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='Ni-HHTP', gas="H2S", ppm=5, rep_ids=[1], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='Zn-HHTP', gas="H2S", ppm=5, rep_ids=[0, 1, 2, 3], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='Zn-HHTP', gas="H2S", ppm=10, rep_ids=[0, 1, 2, 3], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='Zn-HHTP', gas="SO2", ppm=5, rep_ids=[0, 1, 2, 3], time_adjust=150)
        make_adjustment(data, MOF='Cu-HHTP', gas="SO2", ppm=10, rep_ids=[3], time_adjust=200)
        make_adjustment(data, MOF='Cu-HHTP', gas="SO2", ppm=5, rep_ids=[3], n_partitions_slope_adj=5)
        make_adjustment(data, MOF='Cu-HHTP', gas="H2S", ppm=10, rep_ids=[0, 1, 2], time_adjust=200)
        make_adjustment(data, MOF='Cu-HHTP', gas="H2S", ppm=20, rep_ids=[0, 1, 2, 3], time_adjust=200)
        make_adjustment(data, MOF='Cu-HHTP', gas="CO", ppm=5, rep_ids=[0, 1, 2, 3], time_adjust=500)
        make_adjustment(data, MOF='Cu-HHTP', gas="CO", ppm=10, rep_ids=[0, 1, 2, 3], time_adjust=150)
        make_adjustment(data, MOF='Cu-HHTP', gas="CO", ppm=20, rep_ids=[0, 1, 2, 3, 4], time_adjust=100)
        make_adjustment(data, MOF='Cu-HHTP', gas="CO", ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=50)
        make_adjustment(data, MOF='Cu-HHTP', gas="CO", ppm=80, rep_ids=[0, 1, 2, 3], time_adjust=100)
        make_adjustment(data, MOF='Cu-HHTP', gas="NH3", ppm=5, rep_ids=[0, 1, 2, 3], time_adjust=350)
        make_adjustment(data, MOF='Cu-HHTP', gas="NH3", ppm=10, rep_ids=[0, 1, 2], time_adjust=100)
        make_adjustment(data, MOF='Cu-HHTP', gas="NH3", ppm=20, rep_ids=[0, 1, 2, 3], time_adjust=50)
        make_adjustment(data, MOF='Ni-HHTP', gas="NH3", ppm=5, rep_ids=[0, 1, 2, 3], time_adjust=200)
        make_adjustment(data, MOF='Ni-HHTP', gas="NH3", ppm=10, rep_ids=[0, 1, 2], time_adjust=100)
        make_adjustment(data, MOF='Ni-HHTP', gas="NH3", ppm=80, rep_ids=[0, 1, 2, 3], time_adjust=100)
        make_adjustment(data, MOF='Zn-HHTP', gas="NH3", ppm=5, rep_ids=[0, 1, 2, 3], time_adjust=200)
        make_adjustment(data, MOF='Zn-HHTP', gas="NH3", ppm=10, rep_ids=[0, 1, 2, 3], time_adjust=100)
        make_adjustment(data, MOF='Zn-HHTP', gas="NH3", ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=100)
        make_adjustment(data, MOF='Zn-HHTP', gas="CO", ppm=10, rep_ids=[0], n_partitions_slope_adj=5)
        make_adjustment(data, MOF='Zn-HHTP', gas="CO", ppm=20, rep_ids=[0], time_adjust=300, n_partitions_slope_adj=5)
        make_adjustment(data, MOF='Zn-HHTP', gas="CO", ppm=40, rep_ids=[0, 2, 3], time_adjust=200)
        make_adjustment(data, MOF='Zn-HHTP', gas="CO", ppm=80, rep_ids=[1, 2, 3], time_adjust=50)

          # save
        data.to_csv("pure_gases_responses.csv")
    else:
        data = pd.read_csv("pure_gases_responses.csv") # this is adjusted.
        data.drop(columns=['Unnamed: 0'], inplace=True) # remove index column, artifact of reading in
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Assemble and standardize complete array response vectors""")
    return


@app.cell
def _(MOFs, features):
    feature_col_names = [MOF + " " + feature for MOF in MOFs for feature in features]
    return (feature_col_names,)


@app.cell
def _(data):
    data
    return


@app.cell
def _(MOFs, feature_col_names, features, gases, np, pd, ppms):
    def assemble_array_response(data, gases=gases, ppms=ppms, MOFs=MOFs, n_replicates=7, features=features, 
                                feature_col_names=feature_col_names):
        #  matrix will store response features.
        #  col = sensor array response vector
        #  row = particular response feature for a particular MOF (9. 3 MOFs x 3 feature each)
        #  loop through data to build matrix column by column (technically row by row and then transpose)

        matrix = []
        experiments = [] # List which will store experiment setup for each array column
        for gas in gases:
            for ppm in ppms:
                for rep in range(n_replicates):
                    col = []
                    experiment = {'ppm': ppm,
                                'rep_id': rep,
                                 'gas' : gas}
                    for MOF in MOFs:
                        for (i, feature) in enumerate(features):
                            try:
                                val = data.loc[(data['MOF']==MOF)
                                                & (data['gas']==gas)
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


        return combo_df
    return (assemble_array_response,)


@app.cell
def _(assemble_array_response, data):
    combo_df = assemble_array_response(data)
    return (combo_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<!-- # create heatmap of sensor feature values -->""")
    return


@app.cell
def _(PowerTransformer, combo_df, feature_col_names):
    transformed_combo_df = combo_df.copy()
    transformed_combo_df[feature_col_names] = PowerTransformer().fit_transform(transformed_combo_df[feature_col_names])
    transformed_combo_df
    return (transformed_combo_df,)


@app.cell
def _(
    LinearSegmentedColormap,
    feature_col_names,
    gas_to_color,
    make_axes_locatable,
    np,
    plt,
    sns,
):
    def plot_heatmap(transformed_combo_df):
        RdGn = cmap = LinearSegmentedColormap.from_list("mycmap", ["red", "white", "green"])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 10), gridspec_kw={'height_ratios' : [3, 1]})
        # font size
        fs = 20
        # tick labels for features
        yticklabels = ['area', 'slope', 'saturation'] * 3

        # create heatmap
        heat = sns.heatmap(transformed_combo_df[feature_col_names].T, cmap=RdGn, center=0, yticklabels=yticklabels, vmin=-2, vmax=2,
                         square=True, ax=ax1, cbar=False)
        ax1.set_ylabel("response feature", fontsize=fs)

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

        ax1.annotate('SO2', color='black', xy=(19/178, 1.08), xycoords='axes fraction',
                    fontsize=fs, ha='center', va='bottom',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle='-[, widthB=7.8, lengthB=.5', lw=2, color='k'))

        text = ax1.annotate('H$_2$S', color='black', xy=((19 + 9)/89, 1.08), xycoords='axes fraction', 
                    fontsize=fs, ha='center', va='bottom', zorder=1,
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle='-[, widthB=7.5, lengthB=.5', lw=2, color='k'))

        ax1.annotate('NO', color='black', xy=((28 + 18)/89, 1.08), xycoords='axes fraction',
                    fontsize=fs, ha='center', va='bottom',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle='-[, widthB=7.5, lengthB=.5', lw=2, color='k'))

        ax1.annotate('CO', color='black', xy=((46 + 16.5)/89, 1.08), xycoords='axes fraction',
                    fontsize=fs, ha='center', va='bottom', 
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle='-[, widthB=6.25, lengthB=.5', lw=2, color='k'))

        ax1.annotate('NH$_3$', color='black', xy=((62.5+17)/89, 1.08), xycoords='axes fraction',
                    fontsize=fs, ha='center', va='bottom',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle='-[, widthB=7.8, lengthB=.5', lw=2, color='k'))

        # label the MOFs:
        ax1.annotate('Cu', xy=(1.01, 7.5/9), xycoords='axes fraction',
                    fontsize=fs, ha='left', va='center',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=']- ,widthA=1.2, lengthA=1, angleA=180', lw=2, color='k'))

        ax1.annotate('Ni', xy=(1.01, 4.5/9), xycoords='axes fraction',
                    fontsize=fs, ha='left', va='center',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=']- ,widthA=1.2, lengthA=1, angleA=180', lw=2, color='k'))

        ax1.annotate('Zn', xy=(1.01, 1.5/9), xycoords='axes fraction',
                    fontsize=fs, ha='left', va='center',
                    bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                    arrowprops=dict(arrowstyle=']- ,widthA=1.2, lengthA=1, angleA=180', lw=2, color='k'))

        ax1.set_xticks([])
        ax1.set_yticks(ax1.get_yticks())
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=fs)

        colorlist = [gas_to_color[gas] for gas in transformed_combo_df["gas"]] # create list to assign color to each ppm data point

        ax2.grid(axis='x', color='grey', zorder=0)
        ax2.set_xticks(ticks=np.arange(0, len(transformed_combo_df['ppm']), 1), labels=[])
        ax2.tick_params(direction="in", length=10)
        ax2.set_axisbelow(True) 
        ax2.scatter(x=np.arange(0, len(transformed_combo_df['ppm']), 1), y=transformed_combo_df['ppm'], 
                     edgecolor="black", clip_on=False, s=180, c=colorlist)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylabel("concentration\n[ppm]", fontsize=fs)
        ax2.tick_params(axis='both', which='both', labelsize=fs)

        # adjust the position of ax2 to align with ax1
        plt.subplots_adjust(hspace=-0.5)
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax2.set_position([pos1.x0, pos2.y0, pos1.width-0.035, pos2.height - 0.05])
        
        # make ppm plot nice
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_xlim(left=-0.22)
        ax2.set_ylim(top=90, bottom=-0.1)

        ax2.set_yticks(ticks=[80, 40, 0])
        plt.savefig("heatmap_pure_gas.pdf", bbox_inches='tight', pad_inches=0.5)
        plt.show()
    return (plot_heatmap,)


@app.cell
def _(plot_heatmap, transformed_combo_df):
    plot_heatmap(transformed_combo_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Principal Component Analysis (PCA)""")
    return


@app.cell
def _(PCA, combo_df, feature_col_names, pd, transformed_combo_df):
    pcadata = transformed_combo_df[feature_col_names].copy()

    pca = PCA(n_components=2)
    latent_vectors = pca.fit_transform(pcadata)
    z1, z2 = pca.explained_variance_ratio_
    print(z1, z2)

    pcs = pd.DataFrame(data=latent_vectors, columns = ['PC1', 'PC2'])
    pcs_and_exps = pd.concat([combo_df, pcs], axis = 1)
    return latent_vectors, pca, pcadata, pcs, pcs_and_exps, z1, z2


@app.cell
def _(Line2D, gas_to_color, np, plt):
    def plot_PCA(pcs_and_exps, z1, z2, savename="PCA-pure_gases.pdf"):
        pc1 = pcs_and_exps['PC1']
        pc2 = pcs_and_exps['PC2']
        gas = pcs_and_exps['gas']
        ppm = pcs_and_exps['ppm']

        fig, ax = plt.subplots()
        ax.axhline(y=0, color='grey', zorder=0)
        ax.axvline(x=0, color='grey', zorder=0)

        gas_types = [('CO','CO'), ('H2S','H$_2$S'), ('NH3','NH$_3$'), 
                     ('NO','NO'), ('SO2', 'SO$_2$')] # gas label for accessing data and gas label for legend
        ppm_values = pcs_and_exps['ppm'].unique()
        ppm_values.sort()

        # create the bubble plot and legend handles
        gas_legend_elements = []
        for gas_type, gas_label in gas_types:
            gas_mask = (gas == gas_type)
            scatter = ax.scatter(pc1[gas_mask], pc2[gas_mask], s=ppm[gas_mask],
                                edgecolors="black", linewidths=1.5, facecolors=gas_to_color[gas_type])
            gas_legend_elements.append(Line2D([0], [0], marker='o', color='w', label=gas_label,
                                            markeredgecolor="black", markerfacecolor=gas_to_color[gas_type], markersize=10))

        ppm_legend_elements = [Line2D([0], [0], marker='o', color='w', label=str(ppm_value)+" ppm",
                                markerfacecolor='w', markeredgecolor='black', ms=np.sqrt(ppm_value)) for ppm_value in ppm_values]

        # set x and y axis labels and limits
        ax.set_xlabel(f'PC1 score, z$_1$ [{round(z1*100, 1)}%]')
        ax.set_ylabel(f'PC2 score, z$_2$ [{round(z2*100, 1)}%]')
        ax.grid(False)

        # create the legends
        gas_legend = ax.legend(handles=gas_legend_elements, title=None, loc=(1.0,.3), frameon=False)
        ppm_legend = ax.legend(handles=ppm_legend_elements, title=None, loc=(-0.05,-0.4),
                            ncol=len(ppm_values), frameon=False)

        ax.add_artist(gas_legend)
        # ax.add_artist(ppm_legend)
        plt.axis('scaled')
        plt.tight_layout()

        # Adjust the layout
        plt.savefig(savename, bbox_extra_artists=(gas_legend, ppm_legend), bbox_inches='tight') 
        return plt.show()
    return (plot_PCA,)


@app.cell
def _(pcs_and_exps, plot_PCA, plt, z1, z2):
    with plt.rc_context({'font.size': 10}):
        plot_PCA(pcs_and_exps, z1, z2)
    return


if __name__ == "__main__":
    app.run()
