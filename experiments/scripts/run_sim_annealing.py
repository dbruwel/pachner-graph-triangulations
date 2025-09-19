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
        seed="cMcabbgqs",
        gamma_=0.2,
        itts=10_000,
        steps=1,
        lambda_=1e-4,
        alpha=1e-4,
        target_acceptance=0.2,
        potential=potential,
    )

    isos_lists = res["isos_lists"]
    betas_lists = res["betas_lists"]
    acceptances_lists = res["acceptances_lists"]
    final_scores_dict = res["final_scores"]
    final_pns_dict = res["final_pns"]
    final_counts_unknotted_dict = res["final_counts_unknotted"]

    print("\n--- Results ---")
    for i, isos in enumerate(isos_lists):
        if isos:
            print(f"Chain {i+1} ran for {len(isos)} iterations.")
        else:
            print(f"Chain {i+1} did not run successfully.")

    print(f"\nFinal shared scores dictionary has {len(final_scores_dict)} entries.")
    print(f"Final shared pns dictionary has {len(final_pns_dict)} entries.")

    pd.DataFrame.from_dict(final_scores_dict, orient="index", columns=["score"]).to_csv(
        f"{path}/scores.csv"
    )
    pd.DataFrame.from_dict(final_pns_dict, orient="index", columns=["pn"]).to_csv(
        f"{path}/pns.csv"
    )
    pd.DataFrame.from_dict(
        final_counts_unknotted_dict, orient="index", columns=["count_unknotted"]
    ).to_csv(f"{path}/counts_unknotted.csv")
    pd.DataFrame(isos_lists).T.to_csv(f"{path}/isos.csv", index=False)
    pd.DataFrame(betas_lists).T.to_csv(f"{path}/betas.csv", index=False)
    pd.DataFrame(acceptances_lists).T.to_csv(f"{path}/acceptances.csv", index=False)
