import pandas as pd
import os
from datetime import datetime
from multiprocessing import Manager, Process
from pachner_traversal.simulated_annealing import run_chains
from pachner_traversal.potential_functions import edge_degree_variance_potential
import shutil


if __name__ == "__main__":
    src = os.path.abspath(__file__)
    path = "results/" + datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(path, exist_ok=True)
    shutil.copy(src, os.path.join(path, os.path.basename(src)))

    betas_to_run = [1] * 6
    seed = "cMcabbgqs"
    gamma_ = 0.2
    itts = 100_000
    steps = 1
    lambda_ = 1e-4
    alpha = 1e-4
    target_acceptance = 0.2

    (
        isos_lists,
        betas_lists,
        acceptances_lists,
        final_scores_dict,
        final_pns_dict,
        final_counts_unknotted_dict,
    ) = run_chains(
        betas_to_run,
        seed,
        gamma_,
        itts,
        steps,
        lambda_,
        alpha,
        target_acceptance,
        edge_degree_variance_potential,
    )

    print("\n--- Results ---")
    for i, isos in enumerate(isos_lists):
        if isos:
            print(
                f"Chain {i+1} (beta={betas_to_run[i]}) ran for {len(isos)} iterations."
            )
        else:
            print(f"Chain {i+1} (beta={betas_to_run[i]}) did not run successfully.")

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
