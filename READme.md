## Curriculum GEPA

- Initial results on IFEval:

| Model | Budget | Baseline Ref | IFEval Base | GEPA Validation | Curriculum GEPA | Reflective Model* |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Qwen3-0.6B** | - | - | 50.02% | 60.4% / 60.4% | 54.88% / 56.1% | 60.1% / 58.75% |
| **Qwen3-8B** | Medium | Qwen3-0.6B | 49.29% | 59.58% | 56.5% | - |
| **Qwen3-1.7B** | Medium | Qwen3-0.6B | 50.20% | 66.25% | 62.6% | 65% / 63.41% |
| **Gemini-2.5-Flash**| Medium | - | - | - | - | - |

> **Note:** *Reflective Model metrics calculated using `medium=890` metric calls.*