import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from doepy import build
    from scipy.stats import qmc

    import matplotx
    plt.style.use(matplotx.styles.ayu["light"])
    plt.rcParams.update({'font.size': 16, 'lines.linewidth': 3})
    return build, matplotx, mo, np, pd, plt, qmc


@app.cell
def _(pd):
    expts_we_hv = pd.DataFrame(
        {
            'H2S [ppm]': [5, 40, 0, 40, 10, 5, 0, 0, 4, 9, 13, 18, 22, 27, 31, 36, 0, 20],
            'SO2 [ppm]': [5, 40, 40, 0, 0, 0, 10, 5, 36, 13, 27, 4, 18, 31, 9, 22, 20, 0]
        }
    )
    return (expts_we_hv,)


@app.cell
def _():
    ppm_max = 40.0

    design_space = {
        'H2S [ppm]': [0, ppm_max],
        'SO2 [ppm]': [0, ppm_max]
    }
    return design_space, ppm_max


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# the design 🎨""")
    return


@app.cell
def _(build, design_space, np):
    # set seed for reproducibility
    np.random.seed(10)

    # Latin hypercube sampling
    num_samples = 10
    design = build.space_filling_lhs(design_space, num_samples=num_samples)

    # round
    for col in design_space.keys():
        design[col] = design[col].transform(lambda c: np.round(c, 0))

    # write to file
    design.to_csv("concentrations.csv")

    design
    return col, design, num_samples


@app.cell
def _(design, expts_we_hv, plt, ppm_max):
    plt.figure()

    plt.xlabel("H$_2$S [ppm]")
    plt.ylabel("SO$_2$ [ppm]")

    plt.plot([0, ppm_max, ppm_max, 0, 0], [0, 0, ppm_max, ppm_max, 0],
        linestyle="--", color="gray", label="design space", zorder=0
    )

    plt.scatter(design["H2S [ppm]"], design["SO2 [ppm]"], 
        label="planned", edgecolor="C1", facecolor='None', linewidth=3
    )
    plt.scatter(expts_we_hv["H2S [ppm]"], expts_we_hv["SO2 [ppm]"], 
        label="done", marker="s", edgecolor="C2", facecolor='None', linewidth=3
    )

    plt.gca().set_aspect('equal', 'box')

    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # the modified design 🎨

        create a large pool of candidate experiments using a Sobol sequence.
        """
    )
    return


@app.cell
def _(pd, ppm_max, qmc):
    sampler = qmc.Sobol(d=2, scramble=True)

    sobol_samples = sampler.random_base2(m=8) * ppm_max

    sobol_samples = pd.DataFrame(
        sobol_samples, columns=["H2S [ppm]", "SO2 [ppm]"]
    )
    sobol_samples
    return sampler, sobol_samples


@app.cell
def _(plt, sobol_samples):
    plt.figure()
    plt.scatter(
        sobol_samples["H2S [ppm]"], 
        sobol_samples["SO2 [ppm]"], marker="+"
    )
    return


@app.cell
def _(design):
    design
    return


@app.cell
def _(np):
    def get_candidate_max_min_distance_to_data(sobol_samples, design):
        dist_to_data = np.zeros(len(sobol_samples))
        # loop over candidates
        for i in range(len(sobol_samples)):
            # what is the closest data point to this candidate?
            dist_to_data[i] = np.min(
                [np.linalg.norm(
                    design.loc[j, :] - sobol_samples.loc[i, :])
                     for j in range(design.shape[0])]
            )

        new_candidate = np.argmax(dist_to_data)
        return new_candidate
    return (get_candidate_max_min_distance_to_data,)


@app.cell
def _(expts_we_hv, get_candidate_max_min_distance_to_data, sobol_samples):
    new_candidate = get_candidate_max_min_distance_to_data(sobol_samples, expts_we_hv)
    return (new_candidate,)


@app.cell
def _(new_candidate, sobol_samples):
    sobol_samples.loc[new_candidate, :].to_frame().T
    return


@app.cell
def _(get_candidate_max_min_distance_to_data, pd):
    def expand_design(design, n, sobol_samples):
        for i in range(n):
            # get new candidate as one with max min distance to current design
            new_candidate = get_candidate_max_min_distance_to_data(
                sobol_samples, design
            )
            print(new_candidate)
            # add to design, then repeat
            design = pd.concat(
                (design, sobol_samples.loc[new_candidate, :].to_frame().T),
                ignore_index=True
            )
        return design
    return (expand_design,)


@app.cell
def _(sobol_samples):
    sobol_samples
    return


@app.cell
def _(expts_we_hv):
    expts_we_hv
    return


@app.cell
def _(expand_design, expts_we_hv, sobol_samples):
    new_design = expand_design(expts_we_hv, 10, sobol_samples)
    return (new_design,)


@app.cell
def _(new_design):
    new_design
    return


@app.cell
def _(expts_we_hv, new_design, plt, ppm_max):
    plt.figure()

    plt.xlabel("H$_2$S [ppm]")
    plt.ylabel("SO$_2$ [ppm]")

    plt.plot([0, ppm_max, ppm_max, 0, 0], [0, 0, ppm_max, ppm_max, 0],
        linestyle="--", color="gray", label="design space", zorder=0
    )

    plt.scatter(expts_we_hv["H2S [ppm]"], expts_we_hv["SO2 [ppm]"], 
        label="current", edgecolor="C1", linewidth=3
    )
    plt.scatter(
        new_design.loc[len(expts_we_hv):, "H2S [ppm]"], 
        new_design.loc[len(expts_we_hv):, "SO2 [ppm]"], 
        label="new", edgecolor="C2", facecolor='None', linewidth=3
    )

    plt.gca().set_aspect('equal', 'box')

    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.gca()
    return


@app.cell
def _(expts_we_hv, new_design):
    new_design.loc[len(expts_we_hv): ]
    return


if __name__ == "__main__":
    app.run()
