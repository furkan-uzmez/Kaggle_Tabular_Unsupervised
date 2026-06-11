# Submission Results -- TPS Jul 2022

Model selection rule: candidates were selected using internal metrics + stability only. Kaggle leaderboard score is recorded only as a post-mortem learning note and is not used to choose or revise models.

## Final Candidates

| Submission | Source Notebook | Algorithm | Preprocess | Key Params | Internal Rationale | Kaggle ARI (post-mortem) |
| --- | --- | --- | --- | --- | --- | --- |
| `submissions/01_dim_reduction_clustering.csv` | `09_final_selection.py` / `07_dim_reduction_clustering` | KMeans on `umap_d2_nn30` | `quantile+ordinal` | `n_clusters=3`, UMAP `n_components=2`, `n_neighbors=30` on a 15,000-row subsample with 5-NN propagation | Highest non-consensus combined signal score; silhouette 0.7625 and stability mean ARI 1.0 in the recorded run | Not submitted yet |
| `submissions/02_consensus.csv` | `08_consensus_ensemble.py` | Consensus | Mixed source label sets | `n_clusters=3`, average linkage on a 5,000-row co-association matrix with 5-NN propagation | Ensemble robustness candidate chosen for comparison, with leave-one-out consensus ARI mean 0.6326 | Not submitted yet |

## Notes

- If a Kaggle ARI is later recorded here, do not change the selected model based on that score. Treat it as feedback for the post-mortem only.
- Several heavy notebooks were executed with bounded profiles for local feasibility; each `runs/*/result.json` records the actual k ranges, subsample sizes, and stability iterations used.
- The consensus result used subsampling + KNN propagation, so its interpretation differs from a full co-association matrix over all 98,000 rows.

## Post-Mortem Questions

1. Did internal metrics rank the submitted candidates in the same order as Kaggle ARI?
2. Which internal signal appeared most misleading?
3. Was bootstrap stability useful as a tie-breaker?
4. Did consensus improve robustness or only average out useful structure?
