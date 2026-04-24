---
type: community
cohesion: 0.11
members: 28
---

# Cross-Validation Pipeline

**Cohesion:** 0.11 - loosely connected
**Members:** 28 nodes

## Members
- [[.__init__()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_models.py
- [[.forward()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_models.py
- [[CLI entry point for model training with Hydra.  Usage     uv run train     uv r]] - rationale - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/cli/train.py
- [[Custom binary classification head.      Args         in_features  size of the]] - rationale - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_models.py
- [[Load pretrained DenseNet-201 and replace its classification head.      Same stru]] - rationale - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_models.py
- [[Load pretrained ResNet-101 and replace its classification head.      Args]] - rationale - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_models.py
- [[PMGHead]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_models.py
- [[Run 5-fold (or n-fold) cross-validation and save per-fold + summary CSVs.      P]] - rationale - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_crossval.py
- [[Run one epoch (train if optimizer is given, eval otherwise).      Returns]] - rationale - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_train.py
- [[Train one fold and return the test metrics at the best-val-loss epoch.      Para]] - rationale - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_train.py
- [[_build_model()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_train.py
- [[_run_epoch()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_train.py
- [[_save_results()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_crossval.py
- [[_select_device()_1]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_train.py
- [[build_densenet201()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_models.py
- [[build_resnet101()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_models.py
- [[crossval.py]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/cli/crossval.py
- [[crossval_cli()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/cli/crossval.py
- [[get_crossval.py]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_crossval.py
- [[get_models.py]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_models.py
- [[get_train.py]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_train.py
- [[main()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/cli/crossval.py
- [[main()_3]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/cli/train.py
- [[run_crossval()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_crossval.py
- [[train()_14]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_train.py
- [[train.py]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/cli/train.py
- [[train_cli()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/cli/train.py
- [[train_one_fold()]] - code - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/models/get_train.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Cross-Validation_Pipeline
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Ablation Study CLI]]
- 1 edge to [[_COMMUNITY_Training Loop Core]]
- 1 edge to [[_COMMUNITY_Evaluation Metrics]]

## Top bridge nodes
- [[_run_epoch()]] - degree 6, connects to 2 communities
- [[build_densenet201()]] - degree 5, connects to 1 community
- [[build_resnet101()]] - degree 5, connects to 1 community