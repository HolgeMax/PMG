---
source_file: "/Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG/src/func/utils/cfg.py"
type: "rationale"
community: "Preprocessing Pipeline Config"
location: "L12"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Preprocessing_Pipeline_Config
---

# Convert Hydra DictConfig to PreprocessingConfig dataclass.

## Connections
- [[BilateralFilterConfig]] - `uses` [INFERRED]
- [[CLAHEConfig]] - `uses` [INFERRED]
- [[CannyConfig]] - `uses` [INFERRED]
- [[NormalizationConfig]] - `uses` [INFERRED]
- [[PreprocessingConfig]] - `uses` [INFERRED]
- [[config_to_preprocessing_config()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Preprocessing_Pipeline_Config