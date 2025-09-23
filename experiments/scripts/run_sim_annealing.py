import pandas as pd

import pachner_traversal.potential_functions as potentials
from pachner_traversal.simulated_annealing import run_chains
from pachner_traversal.utils import results_path

if __name__ == "__main__":
    path = results_path("sim_annealing")

    potential = potentials.Potential(
        potentials.DeterminantAlexanderPolynomial, max_size=30
    ).calc_potential

    res = run_chains(
        betas=[1] * 6,
        potential=potential,
        seed="cMcabbgqs",
        gamma_=0.2,
        itts=10_000,
        steps=1,
        lambda_=1e-4,
        alpha=1e-4,
        target_acceptance=0.2,
    )

    isos_lists = res["isos_lists"]
    betas_lists = res["betas_lists"]
    acceptances_lists = res["acceptances_lists"]
    scores = res["final_scores"]
    pns = res["final_pns"]
    unknotted = res["final_counts_unknotted"]

    scores_df = pd.DataFrame.from_dict(scores, orient="index", columns=["score"])
    pn_df = pd.DataFrame.from_dict(pns, orient="index", columns=["pn"])
    unknot_df = pd.DataFrame.from_dict(unknotted, orient="index", columns=["unknotted"])
    isos_df = pd.DataFrame(isos_lists).T
    betas_df = pd.DataFrame(betas_lists).T
    acceptances_df = pd.DataFrame(acceptances_lists).T

    scores_df.to_csv(f"{path}/scores.csv")
    pn_df.to_csv(f"{path}/pns.csv")
    unknot_df.to_csv(f"{path}/counts_unknotted.csv")
    isos_df.to_csv(f"{path}/isos.csv", index=False)
    betas_df.to_csv(f"{path}/betas.csv", index=False)
    acceptances_df.to_csv(f"{path}/acceptances.csv", index=False)
