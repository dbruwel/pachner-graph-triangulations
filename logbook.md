# Experiment Logbook
The logbook has been started during the second phase of the project, the honours work was not logged in this format.

## [2025-12-09] Setup Transformer on Dehydration.
We've copied and modified the `data_io` script to be able to acccept the new dehydration signatures. The actual model does not change, just the encoding and decoding step.

The encoding of the dehydration signature is strange, the first charecter represents the size, the next few characters are read in pairs and represent if there's new tetrahedra, the next $N+1$ charecters are used to encode gluing locations, and the last $N+1$ to encode gluing permutations. Because we want to interperate the dehydration signature in pairs, we're going to try and do some suffeling. So the first character will be dropped, the next few read in pairs, and then the last $2(N+1)$ will be joined in pairs, one from the "where" and one from the "how" part of the string. The new tetrahedra stuff will be made uppercase so that it is embedded as a different embedding vector (as it represents fundamentally different information).

## [2025-12-09] Tested transformer on dehydration signatures in pairs.

```yaml
model_type: transformer
model_subtype: gpt2-dehydration
params:
  - d_model: 256
  - num_layers: 6
  - num_heads: 8
  - learning_rate: 0.0005
  - num_train_steps: 50000
  - droput: 0.1
save_location: data/results/sgd_models_dehydration/spheres_256emb_6block_8head_13tet/20251209_1036
p_sphere: 0.00045
p_sphere_sem: 0.00015
```

## [2025-12-10] Ran on dehydration signature in pairs with no dropout.
```yaml
model_type: transformer
model_subtype: gpt2-dehydration
params:
  - d_model: 256
  - num_layers: 6
  - num_heads: 8
  - learning_rate: 0.0005
  - num_train_steps: 50000
  - droput: 0.0
save_location: data/results/sgd_models_dehydration/spheres_256emb_6block_8head_13tet/20251210_0845
p_sphere: 0.00025
p_sphere_sem: 0.00011
```

## [2025-12-10] Ran on dehydration signatures in singles with droput.
```yaml
model_type: transformer
model_subtype: gpt2-dehydration
params:
  - d_model: 256
  - num_layers: 6
  - num_heads: 8
  - learning_rate: 0.0005
  - num_train_steps: 50000
  - droput: 0.1
save_location: data/results/sgd_models_dehydration/spheres_256emb_6block_8head_13tet/20251210_2255
p_sphere: 0.00065
p_sphere_sem: 0.00018
```

This worked better than the pair processing, though I think this is due to the fact that the strings are not really pairs in the case, I'd like to try encoding them individually, but distinguishing between the embedding if its character that represents "new or old", "face ID", or "gluing orientation".


## [2025-12-15] Ran on dehydration signatures in singles with extra encoding.
```yaml
model_type: transformer
model_subtype: gpt2-dehydration
params:
  - d_model: 256
  - num_layers: 6
  - num_heads: 8
  - learning_rate: 0.0005
  - num_train_steps: 50000
  - droput: 0.1
save_location: data/results/sgd_models_dehydration/spheres_256emb_6block_8head_13tet/20251213_0904
```
This model did not train for the full 50,000 itterations, but stopped at 12,000. Despite this the training loss curve was almost identical to the singles without the extra encoding, and when the training curve was extrapolated, had the same "minimum entropy", sugesting that the extra encoding didn't provide any genuine value. Considering all methods produced consistent generation efficiencies, and the singles produced comprable minimum entropies, it would seem the best strategy is to generate the strings in singles with droput with no extra stuff going on. This is, to my understanding, close to what Ash has done, though without dropout. Considering the test loss hasn't diverged, this is likely sufficient.

## [2025-12-18] Meeting ideas
- Get MCMC working for graph encodings - using possibly only a dipole move. In Regina ideally.
- Possibly start working on graph diffusion model on "graph encoded manifold (GEM)"

## [2025-04-16] MCMC, DMT, and knottedness on Dunce Hats and unknotted triangulations.
We want to have a think about a few basic questions experimentally.
1. Can we have a triangulation witha a dunce hat, where all edges are unknotted?
2. Can we have a triangulation with a dunce hat where the edge of the duncehat is knotted (the rest don't matter)
3. If we have a triangulation with no knots in it, will there be a non-perfect Morse Matching?

For the first question, the answer is yes - e.g. `AMLzLvLLzMMPLvQQQPcabcdhjkpkoontnrryvytxuvyzzzoacvdnnabccwlcnbowobscfjxta` and a huge set of other options too.
For the second question, initial searches suggest that the edge always has to be unknotted. After running 10_000 itterations of MCMC no examples of no knotted edges were found. This is likely due to contractibility of the Dunce Hat.
For the third question, the answer seems to be yes, there are many. E.g. `CLwPvALLPAAALALMzPPkacedggijmmlonnoptsrwwvyAyBzBBbgtvmmofqnngjjfgdscagcaaceccc`. That being said, most of the MCMC samples have non-perfect morse matchings according to the code I have, so it could just be the case that the code for checking for a perfect matching isn't very good :/

