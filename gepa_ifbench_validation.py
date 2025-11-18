import dspy
import os
import json
import litellm
from datasets import load_dataset
from pydantic import BaseModel, Field
from pathlib import Path
import random
from typing import List, Dict, Any, Union
from gepa_perplexity import SequenceStatsTracker
import traceback
from ifbench_test import ifeval_score
import Levenshtein
import numpy as np

from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.teleprompt.bootstrap_trace import FailedPrediction

class CustomDspyAdapter(DspyAdapter):
    """Override to filter out logprobs and other fields from reflective dataset."""
    
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        from dspy.teleprompt.bootstrap_trace import FailedPrediction
        from dspy.adapters.chat_adapter import ChatAdapter
        from dspy.adapters.types import History
        from dspy.adapters.types.base_type import Type
        
        print("$$$$$$$$$$$$ CUSTOM ADAPTER BEING CALLED $$$$$$$$$$$$$$")
        program = self.build_program(candidate)

        ret_d = {}
        for pred_name in components_to_update:
            module = None
            for name, m in program.named_predictors():
                if name == pred_name:
                    module = m
                    break
            assert module is not None

            items = []
            for data in eval_batch.trajectories or []:
                trace = data["trace"]
                example = data["example"]
                prediction = data["prediction"]
                module_score = data["score"]
                if hasattr(module_score, "score"):
                    module_score = module_score["score"]

                trace_instances = [t for t in trace if t[0].signature.equals(module.signature)]
                if not self.add_format_failure_as_feedback:
                    trace_instances = [t for t in trace_instances if not isinstance(t[2], FailedPrediction)]
                if len(trace_instances) == 0:
                    continue

                selected = None
                for t in trace_instances:
                    if isinstance(t[2], FailedPrediction):
                        selected = t
                        break

                if selected is None:
                    if isinstance(prediction, FailedPrediction):
                        continue
                    selected = self.rng.choice(trace_instances)

                inputs = selected[1]
                outputs = selected[2]

                new_inputs = {}
                new_outputs = {}

                contains_history = False
                history_key_name = None
                for input_key, input_val in inputs.items():
                    if isinstance(input_val, History):
                        contains_history = True
                        assert history_key_name is None
                        history_key_name = input_key

                if contains_history:
                    s = "```json\n"
                    for i, message in enumerate(inputs[history_key_name].messages):
                        s += f"  {i}: {message}\n"
                    s += "```"
                    new_inputs["Context"] = s

                for input_key, input_val in inputs.items():
                    if contains_history and input_key == history_key_name:
                        continue

                    if isinstance(input_val, Type) and self.custom_instruction_proposer is not None:
                        new_inputs[input_key] = input_val
                    else:
                        new_inputs[input_key] = str(input_val)

                if isinstance(outputs, FailedPrediction):
                    s = "Couldn't parse the output as per the expected output format. The model's raw response was:\n"
                    s += "```\n"
                    s += outputs.completion_text + "\n"
                    s += "```\n\n"
                    new_outputs = s
                else:
                    # ✅ FILTER: Exclude logprobs and related fields
                    fields_to_exclude = {
                        'logprobs', 
                        'raw_completion', 
                        'token_logprobs', 
                        'logprob_std', 
                        'logprob_min', 
                        'logprob_max',
                        'tokens',
                        'entropy',
                        'top_token_confidence',
                        'token_count',
                        'perplexity'
                    }
                    
                    for output_key, output_val in outputs.items():
                        if output_key in fields_to_exclude:
                            print(f"⏭️  Filtering out '{output_key}' from reflective dataset")
                            continue
                        new_outputs[output_key] = str(output_val)

                d = {"Inputs": new_inputs, "Generated Outputs": new_outputs}
                if isinstance(outputs, FailedPrediction):
                    adapter = ChatAdapter()
                    structure_instruction = ""
                    for dd in adapter.format(module.signature, [], {}):
                        structure_instruction += dd["role"] + ": " + dd["content"] + "\n"
                    d["Feedback"] = "Your output failed to parse. Follow this structure:\n" + structure_instruction
                else:
                    feedback_fn = self.feedback_map[pred_name]
                    fb = feedback_fn(
                        predictor_output=outputs,
                        predictor_inputs=inputs,
                        module_inputs=example,
                        module_outputs=prediction,
                        captured_trace=trace,
                    )
                    d["Feedback"] = fb["feedback"]
                    if fb["score"] != module_score:
                        if self.warn_on_score_mismatch:
                            print("Score mismatch - using module level score")
                            self.warn_on_score_mismatch = False
                        fb["score"] = module_score

                items.append(d)

            if len(items) == 0:
                continue
            ret_d[pred_name] = items

        if len(ret_d) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d
    
import dspy.teleprompt.gepa.gepa_utils
dspy.teleprompt.gepa.gepa_utils.DspyAdapter = CustomDspyAdapter

from dspy.teleprompt import GEPA


sequence_tracker = SequenceStatsTracker()


# ---------------- 1. CONFIGURATION & ENVIRONMENT SETUP ----------------

# NOTE: vLLM uses an OpenAI-compatible API, so we use OPENAI_API_BASE and OPENAI_API_KEY.
VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_REFLECTION_API_BASE = "http://localhost:8000/v1"
VLLM_API_KEY = "sk-vllm-placeholder" 
GEMINI_API_KEY = "AIzaSyDeiGggyC3hrHTOhnL35j6inxjm8ocdcxU"
VLLM_MODEL = "Qwen/Qwen3-0.6B" 
#REFLECTION_MODEL = "gemini-2.0-flash" 
REFLECTION_MODEL = "Qwen/Qwen3-0.6B"

os.environ["OPENAI_API_BASE"] = VLLM_API_BASE
os.environ["OPENAI_API_KEY"] = VLLM_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY 

# CRITICAL FIX for Structured Output (JSON Parsing):
# This global setting tells LiteLLM (which DSPy uses) to perform client-side 
# validation and parsing, solving the problem where vLLM ignores the schema.
litellm.enable_json_schema_validation = True
litellm.set_verbose = False # Keep verbose off unless debugging

# ---------------- 2. STRUCTURES & SIGNATURE ----------------

class AnyMathsStructuredOutput(BaseModel):
    final_answer: str = Field(
        ..., description="The final numerical answer to the problem (i.e., just the number or fraction, no units, no other text)"
    )
    solution_pad: str = Field(..., description="The step-by-step solution, detailed reasoning, and derivation of the final answer.")

class SolveMath(dspy.Signature):
    """Solve the given mathematical word problem. The final output must be a valid JSON object matching the provided schema."""
    problem = dspy.InputField()
    final_answer = dspy.OutputField(desc=AnyMathsStructuredOutput.model_fields['final_answer'].description)
    solution_pad = dspy.OutputField(desc=AnyMathsStructuredOutput.model_fields['solution_pad'].description)

class RespondInstructions(dspy.Signature):
    """Respond to the prompt making sure to follow all instructions given meticuously"""
    prompt = dspy.InputField()
    response = dspy.OutputField(desc="Response to the prompt, making sure all instructions are followed")

class RespondInstructionsCurriculum(dspy.Signature):
    """Respond to the prompt by strictly adhering to all specified constraints. Follow this structured approach:  \n\n1. **Parse the Prompt**: Identify and extract explicit requirements, including:  \n   - **Word count** (e.g., \"keep responses under 20 words\").  \n   - **Formatting** (e.g., titles in <<double angular brackets>>, bullet points).  \n   - **Punctuation rules** (e.g., \"no commas\", \"all caps allowed only <4 times\").  \n   - **Language constraints** (e.g., \"response must be in Gujarati\").  \n   - **Domain-specific terms** (e.g., \"Zephyra\" in Example 1).  \n\n2. **Compliance Checklist**:  \n   - **Repeat the prompt** exactly as given before your answer (if required).  \n   - **Use detectable formats** (e.g., <<TITLE>> for titles, bullet points).  \n   - **Avoid prohibited punctuation** (e.g., commas, periods, unless explicitly allowed).  \n   - **Verify domain-specific facts** (e.g., if the task includes niche terms, ensure accuracy even if not pre-known).  \n   - **Address all parts of the prompt**, including hidden constraints (e.g., \"the entire reply must be less than 20 words\").  \n\n3. **Final Validation**:  \n   - Include a <<final_check>> tag to confirm all rules are met.  \n   - Ensure the response is fully compliant with both explicit and implicit instructions.  \n\n**Example**:  \nInput: \"Write a 10-word summary in Spanish. No commas. Use <<TITLE>> for the heading.\"  \nOutput:  \n<<TITLE>> Resumen corto sin comas. (10 words)  \n<<final_check>>  \n\n**Critical Notes**:  \n- **Methodical parsing**: Always extract and prioritize constraints in the order of importance (e.g., word limits > formatting > language).  \n- **Edge cases**: If a constraint conflicts (e.g., \"use commas\" and \"no commas\"), resolve by re-evaluating the prompt for clarification.  \n- **Domain verification**: For tasks requiring niche information (e.g., \"Zephyra\" in Example 1), use external validation if the assistant lacks pre-existing knowledge."""
    prompt = dspy.InputField()
    response = dspy.OutputField(desc="Response to the prompt, making sure all instructions are followed")

class RespondInstructionsStandard(dspy.Signature):
    """New Instruction:  \nRespond to the prompt by first repeating it verbatim without modification. Then, generate your answer while strictly adhering to all specified constraints.  \n\n**Key Requirements:**  \n1. **Prompt Repetition:**  \n   - Begin your response by copying the prompt exactly as provided, including any formatting (e.g., brackets, bullet points, or special characters).  \n   - Do not add, remove, or alter any characters before your answer.  \n\n2. **Title Formatting:**  \n   - If required, enclose the title in double angular brackets (<< >>).  \n   - Ensure titles are standalone and not part of the main response.  \n\n3. **Constraint Adherence:**  \n   - Use **all caps for critical claims** but limit capitalized words to **fewer than 4**.  \n   - Avoid overusing specific keywords (e.g., \"future\" appearing more than once).  \n   - If instructed to exclude certain terms (e.g., \"specific,\" \"exact\"), ensure they are omitted entirely.  \n   - If required to use mathematical notation, avoid commas and ensure equations are fully self-contained (e.g., using LaTeX formatting).  \n\n4. **Comprehensive Coverage:**  \n   - Include all requested elements (e.g., \"dream,\" \"fist fighting,\" \"superpower\") in the response.  \n   - If a task requires a structured output (e.g., a plot summary, list, or dialogue), ensure it is fully developed.  \n\n5. **Error Handling:**  \n   - If any constraint is ambiguous, **assume the most literal interpretation** (e.g., \"title\" refers to a standalone phrase, not part of a sentence).  \n   - Avoid generic phrases like \"## completed ##\" or placeholder text; provide a fully realized answer.  \n   - For structured outputs (e.g., bullet points, numbered lists), use markdown headers (e.g., ### SECTION X) to denote sections.  \n\n**Example of Correct Structure:**  \n[Repeating Prompt]  \n<<Title>> [Answer with constraints met].  \n\nFailure to meet even one requirement (e.g., missing a title, violating capitalization limits) results in task failure. Ensure meticulous attention to detail.  \n\n**Additional Notes:**  \n- Always terminate responses with the exact phrase: **\"Is there anything else I can help with?\"**  \n- For mathematical notation, use LaTeX syntax (e.g., $E = mc^2$) and avoid non-ASCII characters.  \n- If the prompt contains nested brackets (e.g., [[example]]), preserve their structure and nesting level.  \n- Avoid markdown formatting except for title brackets and headers.  \n- If the task involves enumeration (e.g., \"top 10\"), use a numbered list with explicit numbering (e.g., 1., 2., 3.)."""
    prompt = dspy.InputField()
    response = dspy.OutputField(desc="Response to the prompt, making sure all instructions are followed")


# ---------------- 3. DATA LOADING (FIXED) ----------------
def init_ifeval_dataset(limit: int = 50):
    try:
        from datasets import load_dataset
    except ImportError:
        print("CRITICAL ERROR: 'datasets' library not found.")
        exit()

    # 1. Load the original dataset
    dataset = load_dataset("google/IFEval")['train']

    # 2. Split the dataset BEFORE cleaning
    split = dataset.train_test_split(test_size=0.30, seed=42)
    val_test_split = split['test'].train_test_split(test_size=0.5, seed=42)

    train_set_hf = split['train']
    val_set_hf = val_test_split['train']
    test_set_hf = val_test_split['test']

    # 3. Define a new function to convert, clean, and create dspy.Examples
    def convert_and_clean_to_dspy_examples(hf_dataset):
        cleaned_examples = []
        
        # Iterate through the Hugging Face dataset as a standard Python list
        for example in hf_dataset:
            # --- Perform the cleaning logic here ---
            cleaned_kwargs_list = []
            for single_kwargs_dict in example['kwargs']:
                cleaned_dict = {k: v for k, v in single_kwargs_dict.items() if v is not None}
                cleaned_kwargs_list.append(cleaned_dict)
            
            # Create a dspy.Example with the CLEANED kwargs
            cleaned_examples.append(
                dspy.Example(
                    key=example["key"],
                    instruction_id_list=example["instruction_id_list"],
                    prompt=example["prompt"],
                    kwargs=cleaned_kwargs_list  # Use the cleaned list
                ).with_inputs("prompt")
            )
        return cleaned_examples

    # 4. Process each split using the new, reliable function
    train_set = convert_and_clean_to_dspy_examples(train_set_hf)[:limit]
    val_set = convert_and_clean_to_dspy_examples(val_set_hf)[:limit]
    test_set = convert_and_clean_to_dspy_examples(test_set_hf)
    test_set = 2*test_set

    print("Debug: Checking the first item's kwargs from the actual training set.")
    print(train_set[0]['kwargs']) # <-- This will now show the cleaned data

    print("--------------------------------------------------")
    print(f"Dataset loaded: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    print(f"Example prompt: {test_set[0]['prompt']}")
    print("--------------------------------------------------")
    
    # ====== DATA LEAKAGE DETECTION ======
    def check_dataset_leakage(train, val, test, verbose=True):
        """Check for data leakage between train, val, and test sets."""
        
        # Extract unique identifiers (using 'key' field)
        train_keys = set(ex['key'] for ex in train)
        val_keys = set(ex['key'] for ex in val)
        test_keys = set(ex['key'] for ex in test)
        
        # Also check by prompt content (in case keys aren't unique)
        train_prompts = set(ex['prompt'] for ex in train)
        val_prompts = set(ex['prompt'] for ex in val)
        test_prompts = set(ex['prompt'] for ex in test)
        
        # Check intersections
        train_val_key_overlap = train_keys & val_keys
        train_test_key_overlap = train_keys & test_keys
        val_test_key_overlap = val_keys & test_keys
        
        train_val_prompt_overlap = train_prompts & val_prompts
        train_test_prompt_overlap = train_prompts & test_prompts
        val_test_prompt_overlap = val_prompts & test_prompts
        
        # Detect duplicates within each set
        train_duplicates = len(train) - len(train_keys)
        val_duplicates = len(val) - len(val_keys)
        test_duplicates = len(test) - len(test_keys)
        
        if verbose:
            print("\n" + "="*60)
            print("DATA LEAKAGE ANALYSIS")
            print("="*60)
            
            print("\n📊 DATASET SIZES:")
            print(f"  Train set: {len(train)} examples ({len(train_keys)} unique keys)")
            print(f"  Val set:   {len(val)} examples ({len(val_keys)} unique keys)")
            print(f"  Test set:  {len(test)} examples ({len(test_keys)} unique keys)")
            
            print("\n🔍 INTERNAL DUPLICATES (within same set):")
            print(f"  Train: {train_duplicates} duplicate(s)")
            print(f"  Val:   {val_duplicates} duplicate(s)")
            print(f"  Test:  {test_duplicates} duplicate(s) {'⚠️  WARNING!' if test_duplicates > 0 else '✅'}")
            
            print("\n🚨 CROSS-SET LEAKAGE (by key):")
            print(f"  Train ∩ Val:  {len(train_val_key_overlap)} overlap(s) {'⚠️' if train_val_key_overlap else '✅'}")
            print(f"  Train ∩ Test: {len(train_test_key_overlap)} overlap(s) {'⚠️' if train_test_key_overlap else '✅'}")
            print(f"  Val ∩ Test:   {len(val_test_key_overlap)} overlap(s) {'⚠️' if val_test_key_overlap else '✅'}")
            
            print("\n🚨 CROSS-SET LEAKAGE (by prompt content):")
            print(f"  Train ∩ Val:  {len(train_val_prompt_overlap)} overlap(s) {'⚠️' if train_val_prompt_overlap else '✅'}")
            print(f"  Train ∩ Test: {len(train_test_prompt_overlap)} overlap(s) {'⚠️' if train_test_prompt_overlap else '✅'}")
            print(f"  Val ∩ Test:   {len(val_test_prompt_overlap)} overlap(s) {'⚠️' if val_test_prompt_overlap else '✅'}")
            
            # Show specific examples if leakage detected
            if train_val_key_overlap:
                print("\n  ⚠️  Example Train-Val overlap keys:", list(train_val_key_overlap)[:3])
            if train_test_key_overlap:
                print("  ⚠️  Example Train-Test overlap keys:", list(train_test_key_overlap)[:3])
            if val_test_key_overlap:
                print("  ⚠️  Example Val-Test overlap keys:", list(val_test_key_overlap)[:3])
            
            # Summary verdict
            print("\n" + "="*60)
            total_leakage = (len(train_val_key_overlap) + len(train_test_key_overlap) + 
                           len(val_test_key_overlap))
            if total_leakage == 0 and test_duplicates == 0:
                print("✅ NO LEAKAGE DETECTED - Splits are clean!")
            else:
                print("⚠️  LEAKAGE DETECTED - Review split strategy!")
                if test_duplicates > 0:
                    print("   → Test set has duplicates (likely from `test_set = 2*test_set`)")
                if total_leakage > 0:
                    print("   → Cross-set contamination detected")
            print("="*60 + "\n")
        
        return {
            'train_val_overlap': len(train_val_key_overlap),
            'train_test_overlap': len(train_test_key_overlap),
            'val_test_overlap': len(val_test_key_overlap),
            'train_duplicates': train_duplicates,
            'val_duplicates': val_duplicates,
            'test_duplicates': test_duplicates,
            'is_clean': total_leakage == 0 and test_duplicates == 0
        }
    
    # Run leakage check
    leakage_report = check_dataset_leakage(train_set, val_set, test_set, verbose=True)

    return train_set, val_set, test_set


def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Scoring metric: Exact match of the final_answer field after cleaning, required by GEPA."""
    
    from ifbench_test import ifeval_score
    
    try:
        score, feedback = ifeval_score(example, prediction.response)
        return dspy.Prediction(score=score, feedback=feedback)
    
    except:
        print("PROBLEM EVALUATING!")
        return dspy.Prediction(score=0, feedback="Impossible to evaluate output, ignore")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GEPA Benchmark with configurable model and dataset")
    parser.add_argument("--model", type=str, default=VLLM_MODEL, 
                       help="Model name for litellm")
    args = parser.parse_args()
    print("\n" + "="*50)
    print("GEPA DSPY/VLLM BENCHMARK START")
    print("="*50)
    
    # Load data
    train_set, val_set, test_set = init_ifeval_dataset(limit=40) #init_jee_dataset(limit=40)

    # Configure LMs (using standard dspy.LM and injecting VLLM flags via extra_kwargs)
    vllm_lm = dspy.LM(
        model=f"openai/{args.model}", 
        api_base=VLLM_API_BASE,
        api_key=VLLM_API_KEY,
        temperature=1.2, 
        max_tokens=16000,
        max_concurrent_requests=4,
        # VLLM stability flags are passed here via extra_kwargs
        # extra_kwargs={
        #     "extra_body": {
        #         "enforce_eager": True,
        #         "disable_custom_all_reduce": True 
        #     },
        # },
        logprobs=True,
        top_logprobs=20,
    )
    # vllm_lm = dspy.LM(
    #     model=f"gemini/{REFLECTION_MODEL}", 
    #     api_key=GEMINI_API_KEY,
    #     temperature=1.2,
    #     max_tokens=16000,
    # )

    reflection_lm = dspy.LM(
        model=f"openai/{REFLECTION_MODEL}", 
        api_base=VLLM_REFLECTION_API_BASE,
        api_key=VLLM_API_KEY,
        temperature=1.2, 
        max_tokens=16000,
        max_concurrent_requests=4,
        # VLLM stability flags are passed here via extra_kwargs
        extra_kwargs={
            "extra_body": {
                "enforce_eager": True,
                "disable_custom_all_reduce": True 
            },
        },
    )
    
    # reflection_lm = dspy.LM(
    #     model=f"gemini/{REFLECTION_MODEL}", 
    #     api_key=GEMINI_API_KEY,
    #     temperature=1.2,
    #     max_tokens=16000,
    # )

    # Configure DSPy globally
    dspy.settings.configure(lm=vllm_lm)


    base_program = dspy.Predict(RespondInstructions)
    
    # Initialize the base program
    standard_program = dspy.Predict(RespondInstructionsStandard)

    curriculum_program = dspy.Predict(RespondInstructionsCurriculum)

    # --- Step 1: Evaluate Baseline ---
    print("\n" + "="*50)
    print("STEP 1: EVALUATING BASE PROMPT (unoptimized)")
    print("="*50)

    baseline_evaluator = dspy.Evaluate(
        devset=test_set, 
        metric=metric_fn, 
        num_threads=4, 
        display_progress=True,
        provide_traceback=True
    )
    
    baseline_score = baseline_evaluator(base_program, devset=test_set)

    standard_score = baseline_evaluator(standard_program, devset=test_set)
    
    curriculum_score = baseline_evaluator(curriculum_program, devset=test_set)
    
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Model: {args.model} via vLLM")
    print(f"Base Test Score:   {baseline_score.score:.4f}")
    print(f"Standard Test Score:   {standard_score.score:.4f}")
    print(f"Curriculum Test Score:  {curriculum_score.score:.4f}")
    print(f"Improvement (Abs.):    {(curriculum_score.score - baseline_score.score):.4f}")
    print("="*50)


