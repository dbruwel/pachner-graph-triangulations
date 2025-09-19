import pandas as pd

import pachner_traversal.potential_functions as potentials
from pachner_traversal.direct_ascent import run_accent
from pachner_traversal.utils import results_path

if __name__ == "__main__":
    path = results_path("direct_ascent")

    potential = potentials.Potential(
        potentials.AverageEdgeDegree, max_size=None
    ).calc_potential

    res = run_accent(
        base_isos=["bkaajj", "bkaajn", "bkaagj"],
        potential=potential,
        beta=1,
        height=10,
    )

    isos_lists = res

    df_isos = pd.DataFrame([isos_list[0] for isos_list in isos_lists])
    df_scores = pd.DataFrame([isos_list[1] for isos_list in isos_lists])
    df_avg_scores = pd.DataFrame([isos_list[2] for isos_list in isos_lists])
    df_p_knotteds = pd.DataFrame([isos_list[3] for isos_list in isos_lists])
    df_count_unknotteds = pd.DataFrame([isos_list[4] for isos_list in isos_lists])
    df_all_knoteds = pd.DataFrame([isos_list[5] for isos_list in isos_lists])

    df_isos.to_csv(path / "isos.csv", index=False)
    df_scores.to_csv(path / "scores.csv", index=False)
    df_avg_scores.to_csv(path / "avg_scores.csv", index=False)
    df_p_knotteds.to_csv(path / "p_knotteds.csv", index=False)
    df_count_unknotteds.to_csv(path / "count_unknotteds.csv", index=False)
    df_all_knoteds.to_csv(path / "all_knoteds.csv", index=False)
