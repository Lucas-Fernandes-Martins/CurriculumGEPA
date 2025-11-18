## Curriculum GEPA

- Initial results on IFEval:

| Base Model | IFEval Base | Val Score (GEPA) | GEPA | Val Score (Curriculum) | GEPA (Curriculum) | Reflective Model | Budget |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Qwen3-0.6B** | 50.02% | 60.4% / 60.4% | 54.88% / 56.1% | 60.1% / 58.75% | 56.1% / 60.16 | **qwen3-8B** | Medium |
| **Qwen3-0.6B** | 49.29% | 59.58% | 56.5% | - | - | **qwen3-1.7B** | Medium |
| **Qwen3-0.6B** | 50.20% | 66.25% | 62.6% | 65% | 63.41% | **gemini-2.5-flash** | Medium |

> **Note:** "Medium" budget corresponds to 890 metric calls.

> **Note:** *Reflective Model metrics calculated using `medium=890` metric calls.*