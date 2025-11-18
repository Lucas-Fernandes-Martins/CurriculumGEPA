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
VLLM_REFLECTION_API_BASE = "http://localhost:8001/v1"
VLLM_API_KEY = "sk-vllm-placeholder"
#Or set with os.environ directly 
GEMINI_API_KEY = "your-api-key"
VLLM_MODEL = "Qwen/Qwen3-0.6B" 
REFLECTION_MODEL = "gemini-2.5-flash" 


litellm.enable_json_schema_validation = True
litellm.set_verbose = False


# Data signatures
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


#Load ifeval dataset
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
    train_set = convert_and_clean_to_dspy_examples(train_set_hf)
    val_set = convert_and_clean_to_dspy_examples(val_set_hf)[:limit]
    test_set = convert_and_clean_to_dspy_examples(test_set_hf)
    test_set = 2*test_set

    print("Debug: Checking the first item's kwargs from the actual training set.")
    print(train_set[0]['kwargs']) # <-- This will now show the cleaned data

    print("--------------------------------------------------")
    print(f"Dataset loaded: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    print(f"Example prompt: {test_set[0]['prompt']}")
    print("--------------------------------------------------")

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


#------------------------- LLM-as-Judge -----------------------------

# Define a signature for feedback
class FeedbackSignature(dspy.Signature):
    """Give feedback on an incorrect solution to a math problem. It might be incorrect due to the reasoning or the answer formatting,
    solution must be formatted correctly. Make sure the llm_answer and the expected_answer match exactly, pay attention to
    formatting. In the feedback, be specific about the errors in the solution, from logical errors to formatting. Give straight to the point feedback
    which might help a language model improve.
    """
    question: str = dspy.InputField()
    solution: str = dspy.InputField()
    expected_answer: str = dspy.InputField()
    llm_answer: str = dspy.InputField()
    feedback: str = dspy.OutputField(desc="Constructive feedback about the solution")


gemini_lm = dspy.LM(
    model="gemini/gemini-2.5-flash",
    temperature=1.0,
    max_tokens=32000,
    api_key=GEMINI_API_KEY
)

# Function to get feedback
def get_llm_feedback(question, solution, llm_answer, expected_answer):
    # Create the feedback module
    feedback_module = dspy.ChainOfThought(
        FeedbackSignature,
        lm=gemini_lm
    )
    
    with dspy.context(lm=gemini_lm):
        # Call the LLM with structured inputs
        result = feedback_module(
            question=question,
            solution=solution,
            expected_answer=expected_answer,
            llm_answer=llm_answer
        )

    print(f"Expected Answer: {expected_answer} llm_answer: {llm_answer} Feedback:", result.feedback)
   
    return result.feedback


def similarity(a, b):
    dist = Levenshtein.distance(a, b)
    return 1 - dist / max(len(a), len(b))

def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Enhanced scoring metric with explicit, LLM-readable feedback."""
    
    prompt_text = example.prompt if hasattr(example, 'prompt') else ""

    try:
        if hasattr(prediction, 'response'):
            response_text = prediction.response
        elif (hasattr(prediction, 'choices') and len(prediction.choices) > 0 and
              hasattr(prediction.choices[0], 'message') and 
              hasattr(prediction.choices[0].message, 'content')):
            response_text = prediction.choices[0].message.content
        else:
            return dspy.Prediction(score=0, feedback="ERROR: No response found.")

        score, raw_feedback = ifeval_score(example, response_text)
        
        # Extract candidate prompt (full instructions from predictor)
        candidate_prompt = ""
        if trace is not None and len(trace) > 0:
            if isinstance(trace[0], tuple) and len(trace[0]) >= 1:
                predictor = trace[0][0]
                candidate_prompt = predictor.signature.instructions if hasattr(predictor, 'signature') else ""
        
        # Track score and prompt (ignore logprobs for now)
        if pred_name or True:
            sequence_tracker.compute_sequence_stats(
                prediction, 
                prompt_text, 
                system_prompt=candidate_prompt,  # Store full candidate prompt
                score=score,  # Track the score from this metric
                predictor_id=None  # Not needed
            )
            
            # Clean up logprobs to avoid bloat
            if hasattr(prediction, 'logprobs'):
                if hasattr(prediction, '_store'):
                    prediction._store.pop("logprobs", None)

        if not pred_name:
            return dspy.Prediction(score=score, feedback=raw_feedback)

        enhanced_feedback = _create_explicit_feedback(example, response_text, score, raw_feedback)
        
        return dspy.Prediction(score=score, feedback=enhanced_feedback)
    
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"PROBLEM EVALUATING: {e}")
        traceback.print_exc()
        return dspy.Prediction(score=0, feedback=f"Evaluation error: {str(e)[:100]}")
    
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"PROBLEM EVALUATING: {e}")
        traceback.print_exc()
        return dspy.Prediction(score=0, feedback=f"Evaluation error: {str(e)[:100]}")
    


def _create_explicit_feedback(example, response, score, raw_feedback):
    """
    Create concise, plain-text feedback that's easy for an LLM to understand.
    Focuses only on what was right and wrong.
    """
    
    feedback_lines = []
    
    # 1. Parse raw feedback to determine which instructions passed/failed
    instruction_scores = _parse_instruction_scores(raw_feedback, example['instruction_id_list'])
    
    # 2. Add header
    if score == 1.0:
        feedback_lines.append("RESULT: ALL INSTRUCTIONS FOLLOWED ✓")
    else:
        feedback_lines.append("RESULT: SOME INSTRUCTIONS FAILED ✗")

    feedback_lines.append("-" * 30)
    feedback_lines.append("ANALYSIS:")
    
    # 3. Analyze each instruction
    for i, instruction_id in enumerate(example['instruction_id_list']):
        kwargs = example['kwargs'][i] if i < len(example['kwargs']) else {}
        passed = instruction_scores.get(instruction_id, False)
        
        # Get CONCISE explanation
        explanation = _get_detailed_explanation(instruction_id, kwargs, response, passed)
        feedback_lines.append(explanation)
    
    # 4. Add summary
    passed_count = sum(1 for passed in instruction_scores.values() if passed)
    total_count = len(example['instruction_id_list'])
    
    feedback_lines.append("-" * 30)
    feedback_lines.append(f"SUMMARY: {passed_count}/{total_count} instructions passed.")
    
    if score < 1.0:
        failed_instructions = [instr for instr, passed in instruction_scores.items() if not passed]
        feedback_lines.append(f"FAILED: {', '.join(failed_instructions)}")
    
    return "\n".join(feedback_lines)


def _parse_instruction_scores(raw_feedback, instruction_ids):
    """Parse raw feedback to determine which instructions passed."""
    scores = {}
    
    for instruction_id in instruction_ids:
        # Simple heuristic: if instruction_id appears with "1.0", it passed
        if instruction_id in raw_feedback:
            # Look for the score after the instruction_id
            lines = raw_feedback.split('\n')
            for line in lines:
                if instruction_id in line and '1.0' in line:
                    scores[instruction_id] = True
                    break
                elif instruction_id in line and '0.0' in line:
                    scores[instruction_id] = False
                    break
            
            # If not found in lines, default to False
            if instruction_id not in scores:
                scores[instruction_id] = False
        else:
            scores[instruction_id] = False
    
    return scores


def _get_detailed_explanation(instruction_id, kwargs, response, passed):
    """Provide CONCISE, explicit explanations for each constraint type."""
    
    # Word count constraints
    if 'number_words' in instruction_id:
        relation = kwargs.get('relation', 'at least')
        num_words = kwargs.get('num_words', '?')
        actual_words = len(response.split())
        
        if passed:
            return f"✓ Word count: Met requirement ({relation} {num_words}). You wrote {actual_words}."
        else:
            return f"✗ Word count: FAILED. Required: {relation} {num_words}. You wrote: {actual_words}."
    
    # Quotation marks
    elif 'quotation' in instruction_id:
        if passed:
            return "✓ Quotes: Response correctly wrapped in double quotes."
        else:
            issues = []
            if not response.startswith('"'): issues.append("missing opening quote")
            if not response.endswith('"'): issues.append("missing closing quote")
            return f"✗ Quotes: FAILED. Response NOT wrapped in double quotes. Missing: {' & '.join(issues)}."
    
    # No commas
    elif 'no_comma' in instruction_id:
        if passed:
            return "✓ No commas: Response correctly contains 0 commas."
        else:
            comma_count = response.count(',')
            return f"✗ No commas: FAILED. Response contains {comma_count} comma(s). You MUST remove all commas."
    
    # Highlighted sections
    elif 'highlighted_sections' in instruction_id or 'number_highlighted' in instruction_id:
        num_required = kwargs.get('num_highlights', '?')
        num_actual = response.count('*') // 2  # Count pairs
        
        if passed:
            return f"✓ Highlights: Correct number ({num_actual} / {num_required}) using *text* format."
        else:
            return f"✗ Highlights: FAILED. Wrong number of highlights. Required: {num_required}. You provided: {num_actual}. Use *text* to highlight."
    
    # Keywords
    elif 'keywords' in instruction_id:
        required_keywords = kwargs.get('keywords', [])
        response_lower = response.lower()
        missing_keywords = [kw for kw in required_keywords if kw.lower() not in response_lower]
        
        if passed:
            return f"✓ Keywords: All required keywords found ({', '.join(required_keywords)})."
        else:
            return f"✗ Keywords: FAILED. Missing {len(missing_keywords)} keyword(s): {', '.join(missing_keywords)}."
    
    # Number of sentences
    elif 'number_sentences' in instruction_id:
        relation = kwargs.get('relation', 'at least')
        num_sentences_required = kwargs.get('num_sentences', '?')
        actual_sentences = len([s for s in response.split('.') if s.strip()]) # Simple sentence count
        
        if passed:
            return f"✓ Sentence count: Met requirement ({relation} {num_sentences_required}). You wrote {actual_sentences}."
        else:
            return f"✗ Sentence count: FAILED. Required: {relation} {num_sentences_required}. You wrote: {actual_sentences}."
    
    # Lowercase only
    elif 'lowercase' in instruction_id:
        if passed:
            return "✓ Lowercase: Response is correctly all lowercase."
        else:
            return "✗ Lowercase: FAILED. Response contains uppercase letters. You MUST use only lowercase."
    
    # End phrase
    elif 'end_phrase' in instruction_id:
        end_phrase = kwargs.get('end_phrase', '')
        if passed:
            return f"✓ End phrase: Response correctly ends with \"{end_phrase}\"."
        else:
            return f"✗ End phrase: FAILED. Response does NOT end with required phrase. Required: \"{end_phrase}\"."
    
    # Constrained response (specific phrases)
    elif 'constrained_response' in instruction_id:
        if passed:
            return "✓ Constrained phrase: Response correctly used one of the required phrases."
        else:
            return "✗ Constrained phrase: FAILED. Response MUST contain 'My answer is yes', 'My answer is no', or 'My answer is maybe'."
    
    # Repeat prompt
    elif 'repeat_prompt' in instruction_id:
        if passed:
            return "✓ Repeat prompt: Prompt was correctly repeated."
        else:
            return "✗ Repeat prompt: FAILED. You MUST repeat the prompt exactly before your answer."
    
    # Default fallback
    else:
        if passed:
            return f"✓ Constraint '{instruction_id}': Passed."
        else:
            return f"✗ Constraint '{instruction_id}': FAILED. Parameters: {kwargs}."
        
# ---------------- 5. MAIN EXECUTION ----------------

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
        logprobs=True,
        top_logprobs=20,
    )

    eval_lm = vllm_lm.copy(temperature=1.0)

    # vllm_lm = dspy.LM(
    #     model=f"gemini/{REFLECTION_MODEL}", 
    #     api_key=GEMINI_API_KEY,
    #     temperature=1.2,
    #     max_tokens=16000,
    # )

    reflection_lm = dspy.LM(
        model=f"gemini/{REFLECTION_MODEL}", 
        api_key=GEMINI_API_KEY,
        temperature=1.2,
        max_tokens=16000,
    )

    # Configure DSPy globally
    dspy.settings.configure(lm=vllm_lm)
    
    # Initialize the base program
    base_program = dspy.Predict(RespondInstructions)

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
    
    with dspy.context(lm=eval_lm):
        # sequence_tracker.set_stage("baseline")
        baseline_score = baseline_evaluator(base_program, devset=test_set)
        print(f"\nBaseline Test Score: {baseline_score.score:.4f}")

    # --- Step 2: Optimize with GEPA ---
    print("\n" + "="*50)
    print(f"STEP 2: OPTIMIZING PROMPT with (Budget: 500)")
    print("="*50)

    sequence_tracker.set_stage("training") 

    optimizer = GEPA(
        metric=metric_with_feedback,
        reflection_lm=reflection_lm,
        auto="medium", 
        #max_metric_calls=200, 
        num_threads=1, 
        reflection_minibatch_size=3,
        track_stats=True,
    )

    optimized_program = optimizer.compile(
        base_program,
        trainset=train_set,
        valset=val_set,
    )
    
    # --- Step 3: Evaluate Optimized Program ---
    print("\n" + "="*50)
    print("STEP 3: EVALUATING OPTIMIZED PROGRAM")
    print("="*50)

    sequence_tracker.set_stage("test")

    with dspy.context(lm=eval_lm):
        optimized_score = baseline_evaluator(optimized_program, devset=test_set)
    
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Model: {args.model} via vLLM")
    print(f"Baseline Test Score:   {baseline_score.score:.4f}")
    print(f"Optimized Test Score:  {optimized_score.score:.4f}")
    print(f"Improvement (Abs.):    {(optimized_score.score - baseline_score.score):.4f}")
    print("="*50)

        # Add this function at module level (around line 650, before main execution):
    def extract_candidate_data(results):
        """Extract candidates with scores and prompts"""
        candidate_data = []
        
        for i, (candidate, score) in enumerate(zip(results.candidates, results.val_aggregate_scores)):
            for name, predictor in candidate.named_predictors():
                prompt = predictor.signature.instructions
                
                candidate_entry = {
                    'candidate_idx': i,
                    'score': score,
                    'prompt': prompt,
                    'prompt_length_chars': len(prompt),
                    'prompt_length_words': len(prompt.split()),
                    'predictor_name': name
                }
                
                candidate_data.append(candidate_entry)
                break
        
        return candidate_data

# Then at the very end of if __name__ == "__main__": (after line 700+), add this:

    # ============================================================================
    # EXTRACT AND ALIGN PERPLEXITY DATA WITH GEPA CANDIDATES
    # ============================================================================

# Replace the entire alignment section (lines 750-1050) with this simple export:

if not hasattr(optimized_program, 'detailed_results'):
    print("⚠️  WARNING: detailed_results not found.")
else:
    results = optimized_program.detailed_results
    
    print("\n" + "="*80)
    print("💾 EXPORTING CANDIDATE DATA AND SCORES")
    print("="*80)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = './gepa_sequence_stats'
    os.makedirs(results_dir, exist_ok=True)
    
    # ============================================================================
    # DATASET 1: GEPA Candidate Data (from detailed_results)
    # ============================================================================
    
    candidate_data = []
    for i, (candidate, score) in enumerate(zip(results.candidates, results.val_aggregate_scores)):
        for name, predictor in candidate.named_predictors():
            prompt = predictor.signature.instructions
            
            candidate_entry = {
                'candidate_idx': i,
                'score': float(score),
                'prompt': prompt,
                'prompt_length_chars': len(prompt),
                'prompt_length_words': len(prompt.split()),
                'predictor_name': name,
                'is_best': (i == results.best_idx)
            }
            
            candidate_data.append(candidate_entry)
            break
    
    candidates_file = os.path.join(results_dir, f'gepa_candidates_{timestamp}.json')
    with open(candidates_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'total_candidates': len(candidate_data),
                'best_candidate_idx': results.best_idx,
                'best_score': float(results.val_aggregate_scores[results.best_idx])
            },
            'candidates': candidate_data
        }, f, indent=2)
    
    print(f"✅ GEPA candidates saved to: {candidates_file}")
    print(f"   Total candidates: {len(candidate_data)}")
    
    # ============================================================================
    # DATASET 2: Sequence Tracker Data (scores + perplexity per evaluation)
    # ============================================================================
    
    tracker_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_evaluations': len(sequence_tracker.stats),
            'evaluations_with_perplexity': sum(1 for s in sequence_tracker.stats if s.get('perplexity') is not None)
        },
        'evaluations': []
    }
    
    for i, stat in enumerate(sequence_tracker.stats):
        eval_entry = {
            'eval_id': i,
            'candidate_prompt': stat.get('candidate_prompt', '')[:200] + "...",  # First 200 chars for matching
            'candidate_prompt_hash': hash(stat.get('candidate_prompt', '')),  # Hash for easy matching
            'user_prompt': stat.get('user_prompt', ''),
            'score': stat.get('score'),
            'perplexity': stat.get('perplexity'),
            'confidence': stat.get('top_token_confidence'),
            'token_count': stat.get('token_count'),
            'stage': stat.get('stage'),
            'timestamp': stat.get('timestamp')
        }
        tracker_data['evaluations'].append(eval_entry)
    
    tracker_file = os.path.join(results_dir, f'sequence_tracker_data_{timestamp}.json')
    with open(tracker_file, 'w') as f:
        json.dump(tracker_data, f, indent=2)
    
    print(f"✅ Sequence tracker data saved to: {tracker_file}")
    print(f"   Total evaluations: {len(sequence_tracker.stats)}")
    
    # ============================================================================
    # DATASET 3: Summary Statistics
    # ============================================================================
    
    # Calculate summary stats per unique prompt
    from collections import defaultdict
    prompt_stats = defaultdict(lambda: {
        'scores': [],
        'perplexities': [],
        'confidences': [],
        'count': 0
    })
    
    for stat in sequence_tracker.stats:
        prompt_hash = hash(stat.get('candidate_prompt', ''))
        prompt_stats[prompt_hash]['scores'].append(stat.get('score'))
        prompt_stats[prompt_hash]['count'] += 1
        if stat.get('perplexity'):
            prompt_stats[prompt_hash]['perplexities'].append(stat['perplexity'])
        if stat.get('top_token_confidence'):
            prompt_stats[prompt_hash]['confidences'].append(stat['top_token_confidence'])
    
    summary_data = {
        'metadata': {
            'timestamp': timestamp,
            'unique_prompts': len(prompt_stats)
        },
        'prompt_statistics': []
    }
    
    for prompt_hash, stats in prompt_stats.items():
        summary_entry = {
            'prompt_hash': prompt_hash,
            'num_evaluations': stats['count'],
            'mean_score': float(np.mean(stats['scores'])) if stats['scores'] else None,
            'std_score': float(np.std(stats['scores'])) if stats['scores'] else None,
            'mean_perplexity': float(np.mean(stats['perplexities'])) if stats['perplexities'] else None,
            'std_perplexity': float(np.std(stats['perplexities'])) if stats['perplexities'] else None,
            'mean_confidence': float(np.mean(stats['confidences'])) if stats['confidences'] else None,
        }
        summary_data['prompt_statistics'].append(summary_entry)
    
    summary_file = os.path.join(results_dir, f'summary_statistics_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"✅ Summary statistics saved to: {summary_file}")
    print(f"   Unique prompts tracked: {len(prompt_stats)}")
    
    # ============================================================================
    # Print Summary
    # ============================================================================
    
    print(f"\n{'='*80}")
    print("📊 EXPORT SUMMARY")
    print(f"{'='*80}")
    print(f"✅ 3 files saved:")
    print(f"   1. {candidates_file}")
    print(f"   2. {tracker_file}")
    print(f"   3. {summary_file}")
    print(f"\n💡 To match candidates with scores:")
    print(f"   - Use 'candidate_prompt_hash' to match tracker evaluations")
    print(f"   - Or use fuzzy string matching on 'candidate_prompt' (first 200 chars)")
    print(f"{'='*80}\n")

    # ✅ NEW: Add sequence statistics summary
    # print("\n" + "="*50)
    # print("📊 SEQUENCE STATISTICS SUMMARY")
    # print("="*50)
        
    # # ✅ Show overall stats
    # print("\n" + "="*50)
    # print("📊 SEQUENCE STATISTICS SUMMARY")
    # print("="*50)

    # total_stats = len(sequence_tracker.stats)
    # stats_with_perplexity = sum(1 for s in sequence_tracker.stats if s['perplexity'] is not None)

    # print(f"\n📊 Overall Statistics:")
    # print(f"   Total samples tracked: {total_stats}")
    # print(f"   Samples with perplexity: {stats_with_perplexity}")

    # if stats_with_perplexity > 0:
    #     all_perplexities = [s['perplexity'] for s in sequence_tracker.stats if s['perplexity'] is not None]
    #     all_confidences = [s['top_token_confidence'] for s in sequence_tracker.stats if s['top_token_confidence'] is not None]
        
    #     print(f"   Mean Perplexity: {np.mean(all_perplexities):.2f} ± {np.std(all_perplexities):.2f}")
    #     print(f"   Perplexity Range: [{min(all_perplexities):.2f}, {max(all_perplexities):.2f}]")
    #     if all_confidences:
    #         print(f"   Mean Confidence: {np.mean(all_confidences):.4f} ± {np.std(all_confidences):.4f}")

    # print("\n" + "="*50)
    
    # # ✅ Save statistics and create visualizations
    # results_dir = './gepa_sequence_stats'
    # os.makedirs(results_dir, exist_ok=True)    
    # from datetime import datetime
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # # Save JSON statistics
    # stats_json_file = os.path.join(results_dir, f'sequence_stats_{timestamp}.json')
    # sequence_tracker.save_stats(stats_json_file)
    
    # # Create visualizations
    # plot_file = os.path.join(results_dir, f'sequence_stats_plot_{timestamp}.png')
    # sequence_tracker.plot_stats(plot_file)
    
    # # ✅ Print detailed historical data
    # print(f"\n📋 DETAILED HISTORICAL DATA ({len(sequence_tracker.stats)} total samples):")
    # print("-" * 100)
    # print(f"{'#':<5} {'Stage':<12} {'Perplexity':<15} {'Confidence':<15} {'Tokens':<10} {'Prompt':<50}")
    # print("-" * 100)
    
    # for i, stat in enumerate(sequence_tracker.stats):
    #     perp_str = f"{stat['perplexity']:.2f}" if stat['perplexity'] is not None else "N/A"
    #     conf_str = f"{stat['top_token_confidence']:.4f}" if stat['top_token_confidence'] is not None else "N/A"
    #     tokens_str = f"{stat['token_count']}"
    #     prompt_preview = stat['prompt'][:40] + "..." if len(stat['prompt']) > 40 else stat['prompt']
        
    #     print(f"{i:<5} {stat['stage']:<12} {perp_str:<15} {conf_str:<15} {tokens_str:<10} {prompt_preview:<50}")
    
    # print("-" * 100)
    # print(f"\n✅ Analysis complete!")
    # print(f"   📁 Statistics saved to: {stats_json_file}")
    # print(f"   📊 Plot saved to: {plot_file}")
    # print(f"\n   To view plots, open: {plot_file}")


    # # ✅ NEW: Per-candidate statistics with SCORE and per-question breakdown
    # print("\n" + "="*50)
    # print("📊 PER-CANDIDATE STATISTICS (RANKED BY SCORE)")
    # print("="*50)
    
    # candidate_stats = sequence_tracker.get_candidate_summary()
    # if candidate_stats:
    #     print(f"\nFound statistics for {len(candidate_stats)} unique candidates\n")
        
    #     # Sort by score (highest first = best candidates)
    #     sorted_candidates = sorted(
    #         candidate_stats.items(),
    #         key=lambda x: x[1]['mean_score'] if x[1]['mean_score'] else 0,
    #         reverse=True
    #     )
        
    #     print(f"{'#':<3} {'Candidate':<12} {'Mean Score':<12} {'Mean PPL':<12} {'Confidence':<15} {'Avg Q Score':<15} {'#Q':<6} {'Prompt Preview':<40}")
    #     print("-" * 140)
        
    #     for i, (cand_id, stats) in enumerate(sorted_candidates, 1):
    #         score = stats['mean_score'] if stats['mean_score'] else 0
    #         ppl = stats['mean_perplexity']
    #         conf = stats['mean_confidence'] if stats['mean_confidence'] else 0
            
    #         # ✅ NEW: Calculate average of per-question scores
    #         if stats['avg_score_per_question']:
    #             avg_q_score = np.mean(list(stats['avg_score_per_question'].values()))
    #         else:
    #             avg_q_score = 0
            
    #         num_questions = stats['num_unique_questions']
    #         preview = stats['system_prompt'][:30] + "..."
            
    #         print(f"{i:<3} {cand_id:<12} {score:<12.4f} {ppl:<12.2f} {conf:<15.4f} {avg_q_score:<15.4f} {num_questions:<6} {preview:<40}")
        
    #     # ✅ NEW: Detailed per-question breakdown
    #     print("\n" + "="*50)
    #     print("📋 DETAILED PER-QUESTION SCORE BREAKDOWN")
    #     print("="*50)
        
    #     for i, (cand_id, stats) in enumerate(sorted_candidates[:3], 1):  # Top 3 candidates
    #         print(f"\n🔹 Candidate {cand_id} (Mean Score: {stats['mean_score']:.4f})")
    #         print(f"   System Prompt: {stats['system_prompt_full'][:100]}...")
    #         print(f"   {'Question':<60} {'Avg Score':<12}")
    #         print(f"   {'-'*72}")
            
    #         # Sort questions by score (highest first)
    #         sorted_questions = sorted(
    #             stats['avg_score_per_question'].items(),
    #             key=lambda x: x[1],
    #             reverse=True
    #         )
            
    #         for question, q_score in sorted_questions[:5]:  # Show top 5 questions per candidate
    #             q_preview = question[:55] + "..." if len(question) > 55 else question
    #             print(f"   {q_preview:<60} {q_score:<12.4f}")
    
    # print("\n" + "="*50)
    
    # # ✅ Save per-candidate statistics
    # candidate_stats_json = os.path.join(results_dir, f'per_candidate_stats_{timestamp}.json')
    # sequence_tracker.save_candidate_stats(candidate_stats_json)
    
    # # ✅ Create per-candidate visualization
    # candidate_plot_file = os.path.join(results_dir, f'per_candidate_stats_{timestamp}.png')
    # sequence_tracker.plot_candidate_stats(candidate_plot_file, top_n=10)
    
    # print(f"\n✅ Analysis complete!")
    # print(f"   📁 Per-candidate statistics: {candidate_stats_json}")
    # print(f"   📊 Per-candidate plots: {candidate_plot_file}")


    # # Add this at the end of the main execution block, after GEPA optimization completes

    # # ============================================================================
    # # ✅ EXTRACT GEPA CANDIDATE AND SCORE TREE
    # # ============================================================================
    
    # print("\n" + "="*80)
    # print("📊 EXTRACTING GEPA EVOLUTION TREE AND CANDIDATES")
    # print("="*80)
    
    # # Check if detailed_results are available
    # if not hasattr(optimized_program, 'detailed_results'):
    #     print("⚠️  WARNING: detailed_results not found. Make sure GEPA was run with track_stats=True")
    # else:
    #     results = optimized_program.detailed_results
        
    #     # -------- Part 1: Extract basic candidate data --------
    #     def extract_candidate_data(results):
    #         """Extract candidates with scores and prompts"""
    #         candidate_data = []
            
    #         for i, (candidate, score) in enumerate(zip(results.candidates, results.val_aggregate_scores)):
    #             for name, predictor in candidate.named_predictors():
    #                 prompt = predictor.signature.instructions
                    
    #                 candidate_entry = {
    #                     'candidate_idx': i,
    #                     'score': score,
    #                     'prompt': prompt,
    #                     'prompt_length_chars': len(prompt),
    #                     'prompt_length_words': len(prompt.split()),
    #                     'predictor_name': name
    #                 }
                    
    #                 candidate_data.append(candidate_entry)
    #                 break  # Just take first predictor for each candidate
            
    #         return candidate_data
        
    #     candidate_data = extract_candidate_data(results)
    #     print(f"✅ Extracted {len(candidate_data)} candidates")
        
    #     # -------- Part 2: Build evolution tree with ancestry --------
    #     def build_evolution_tree(candidates, parents, results):
    #         """Build evolution tree with parent-child relationships"""
    #         tree = {
    #             'nodes': [],
    #             'edges': [],
    #             'levels': {},
    #             'roots': []
    #         }
            
    #         for i, candidate in enumerate(candidates):
    #             parent_list = parents[i] if i < len(parents) else [None]
    #             parent_idx = parent_list[0] if parent_list and parent_list[0] is not None else None
                
    #             node = {
    #                 'id': i,
    #                 'candidate_idx': candidate['candidate_idx'],
    #                 'parent_idx': parent_idx,
    #                 'score': candidate['score'],
    #                 'prompt': candidate['prompt'],
    #                 'prompt_length_chars': candidate['prompt_length_chars'],
    #                 'prompt_length_words': candidate['prompt_length_words'],
    #                 'predictor_name': candidate['predictor_name'],
    #                 'children': [],
    #                 'level': 0,
    #                 'is_best': i == results.best_idx
    #             }
                
    #             if parent_idx is None:
    #                 node['level'] = 0
    #                 tree['roots'].append(i)
    #             else:
    #                 parent_level = next((n['level'] for n in tree['nodes'] if n['id'] == parent_idx), 0)
    #                 node['level'] = parent_level + 1
                
    #             tree['nodes'].append(node)
                
    #             if parent_idx is not None:
    #                 tree['edges'].append({
    #                     'from': parent_idx,
    #                     'to': i,
    #                     'score_delta': candidate['score'] - candidates[parent_idx]['score'] if parent_idx < len(candidates) else 0
    #                 })
            
    #         # Build children lists
    #         for edge in tree['edges']:
    #             parent_node = next(n for n in tree['nodes'] if n['id'] == edge['from'])
    #             parent_node['children'].append(edge['to'])
            
    #         # Group by levels
    #         for node in tree['nodes']:
    #             level = node['level']
    #             if level not in tree['levels']:
    #                 tree['levels'][level] = []
    #             tree['levels'][level].append(node['id'])
            
    #         return tree
        
    #     evolution_tree = build_evolution_tree(candidate_data, results.parents, results)
    #     print(f"✅ Built evolution tree with {len(evolution_tree['nodes'])} nodes")
    #     print(f"   - Root candidates: {len(evolution_tree['roots'])}")
    #     print(f"   - Evolution levels: {max(evolution_tree['levels'].keys()) + 1 if evolution_tree['levels'] else 0}")
        
    #     # -------- Part 3: Print tree visualization --------
    #     print("\n🌳 EVOLUTION TREE VISUALIZATION:")
    #     print("-" * 80)
    #     for level, nodes in sorted(evolution_tree['levels'].items()):
    #         print(f"\n📊 Generation {level}:")
    #         for node_id in nodes:
    #             node = next(n for n in evolution_tree['nodes'] if n['id'] == node_id)
    #             parent_str = f" (parent: {node['parent_idx']})" if node['parent_idx'] is not None else " (root)"
    #             best_marker = " ⭐ BEST" if node['is_best'] else ""
    #             print(f"   ├─ Candidate #{node_id}: score={node['score']:.4f}{parent_str}{best_marker}")
        
    #     # -------- Part 4: Score analysis --------
    #     print("\n\n📈 SCORE PROGRESSION ANALYSIS:")
    #     print("-" * 80)
    #     scores = [c['score'] for c in candidate_data]
    #     print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    #     print(f"Average score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
    #     best_candidate = max(candidate_data, key=lambda x: x['score'])
    #     worst_candidate = min(candidate_data, key=lambda x: x['score'])
        
    #     print(f"\n🏆 Best candidate:")
    #     print(f"   Candidate #{best_candidate['candidate_idx']}: {best_candidate['score']:.4f}")
    #     print(f"   Prompt length: {best_candidate['prompt_length_chars']} chars ({best_candidate['prompt_length_words']} words)")
        
    #     print(f"\n📉 Worst candidate:")
    #     print(f"   Candidate #{worst_candidate['candidate_idx']}: {worst_candidate['score']:.4f}")
    #     print(f"   Prompt length: {worst_candidate['prompt_length_chars']} chars ({worst_candidate['prompt_length_words']} words)")
        
    #     # -------- Part 5: Get lineage to best candidate --------
    #     def get_candidate_lineage(results, candidate_idx):
    #         """Get the path from seed to a specific candidate"""
    #         lineage = [candidate_idx]
    #         current = candidate_idx
            
    #         while True:
    #             parents = results.parents[current]
    #             if not parents or all(p is None for p in parents):
    #                 break
    #             current = next((p for p in parents if p is not None), None)
    #             if current is None:
    #                 break
    #             lineage.insert(0, current)
            
    #         return lineage
        
    #     best_lineage = get_candidate_lineage(results, results.best_idx)
    #     print(f"\n🔗 EVOLUTION PATH (Seed → Best):")
    #     print(f"   {' → '.join([f'#{i}' for i in best_lineage])}")
        
    #     # -------- Part 6: Export to JSON --------
    #     print("\n\n💾 EXPORTING RESULTS TO JSON:")
    #     print("-" * 80)
        
    #     export_data = {
    #         'metadata': {
    #             'total_candidates': len(candidate_data),
    #             'total_generations': len(evolution_tree['levels']),
    #             'best_candidate_idx': results.best_idx,
    #             'best_score': best_candidate['score'],
    #             'worst_score': worst_candidate['score'],
    #             'mean_score': float(np.mean(scores)),
    #             'total_metric_calls': results.total_metric_calls if hasattr(results, 'total_metric_calls') else None,
    #         },
    #         'candidates': candidate_data,
    #         'evolution_tree': {
    #             'nodes': evolution_tree['nodes'],
    #             'edges': evolution_tree['edges'],
    #             'levels': {str(k): v for k, v in evolution_tree['levels'].items()},
    #             'roots': evolution_tree['roots']
    #         },
    #         'best_lineage': best_lineage,
    #         'scores': scores
    #     }
        
    #     # Save to file
    #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #     export_filename = os.path.join(results_dir, f'gepa_ifbench_results_{timestamp}.json')
        
    #     with open(export_filename, 'w') as f:
    #         json.dump(export_data, f, indent=2)
        
    #     print(f"✅ Results exported to: {export_filename}")
        
    #     # -------- Part 7: Print best prompt --------
    #     print("\n\n🎯 BEST OPTIMIZED PROMPT:")
    #     print("=" * 80)
    #     print(best_candidate['prompt'])
    #     print("=" * 80)
        
    #     # -------- Part 8: Create visualizations --------
    #     print("\n\n📊 CREATING VISUALIZATIONS:")
    #     print("-" * 80)
        
    #     import matplotlib.pyplot as plt
        
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
    #     prompt_lengths = [c['prompt_length_chars'] for c in candidate_data]
    #     candidate_nums = list(range(len(candidate_data)))
        
    #     # Plot 1: Score vs Prompt Length
    #     scatter1 = ax1.scatter(prompt_lengths, scores, c=candidate_nums, cmap='viridis', alpha=0.6, s=80)
    #     ax1.scatter([best_candidate['prompt_length_chars']], [best_candidate['score']], 
    #                c='red', s=200, marker='*', edgecolor='black', linewidth=2, label='Best', zorder=5)
    #     ax1.set_xlabel('Prompt Length (characters)', fontsize=12)
    #     ax1.set_ylabel('Score', fontsize=12)
    #     ax1.set_title('Score vs Prompt Length', fontsize=14, fontweight='bold')
    #     ax1.grid(True, alpha=0.3)
    #     ax1.legend()
    #     plt.colorbar(scatter1, ax=ax1, label='Candidate #')
        
    #     # Plot 2: Score Evolution Over Time
    #     ax2.plot(candidate_nums, scores, 'b-o', markersize=6, alpha=0.7, linewidth=2)
    #     ax2.scatter([results.best_idx], [best_candidate['score']], 
    #                c='red', s=200, marker='*', edgecolor='black', linewidth=2, label='Best', zorder=5)
    #     ax2.set_xlabel('Candidate Number (Evolution Order)', fontsize=12)
    #     ax2.set_ylabel('Score', fontsize=12)
    #     ax2.set_title('Score Evolution Over Time', fontsize=14, fontweight='bold')
    #     ax2.grid(True, alpha=0.3)
    #     ax2.legend()
        
    #     # Plot 3: Prompt Length Evolution
    #     ax3.plot(candidate_nums, prompt_lengths, 'g-s', markersize=5, alpha=0.7, linewidth=2)
    #     ax3.scatter([results.best_idx], [best_candidate['prompt_length_chars']], 
    #                c='red', s=200, marker='*', edgecolor='black', linewidth=2, label='Best', zorder=5)
    #     ax3.set_xlabel('Candidate Number', fontsize=12)
    #     ax3.set_ylabel('Prompt Length (characters)', fontsize=12)
    #     ax3.set_title('Prompt Length Evolution', fontsize=14, fontweight='bold')
    #     ax3.grid(True, alpha=0.3)
    #     ax3.legend()
        
    #     # Plot 4: Score Distribution by Generation
    #     generation_scores = []
    #     generation_labels = []
    #     for gen in sorted(evolution_tree['levels'].keys()):
    #         node_ids = evolution_tree['levels'][gen]
    #         gen_scores = [scores[nid] for nid in node_ids]
    #         generation_scores.append(gen_scores)
    #         generation_labels.append(f'Gen {gen}')
        
    #     ax4.boxplot(generation_scores, labels=generation_labels)
    #     ax4.set_ylabel('Score', fontsize=12)
    #     ax4.set_title('Score Distribution by Generation', fontsize=14, fontweight='bold')
    #     ax4.grid(True, alpha=0.3, axis='y')
        
    #     plt.suptitle(f'GEPA Optimization Results (IFEval - {timestamp})', 
    #                 fontsize=16, fontweight='bold', y=0.995)
    #     plt.tight_layout(rect=[0, 0, 1, 0.99])
        
    #     plot_filename = os.path.join(results_dir, f'gepa_ifbench_visualization_{timestamp}.png')
    #     plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    #     print(f"✅ Visualization saved to: {plot_filename}")
    #     plt.close()
        
    #     print(f"\n{'='*80}")
    #     print("✅ GEPA ANALYSIS COMPLETE!")
    #     print(f"{'='*80}")

    # # ... existing GEPA extraction code ...

    # if not hasattr(optimized_program, 'detailed_results'):
    #     print("⚠️  WARNING: detailed_results not found.")
    # else:
    #     results = optimized_program.detailed_results
    #     candidate_data = extract_candidate_data(results)
        
    #     # ✅ ALIGN PERPLEXITY STATS WITH CANDIDATES
    #     matched, unmatched = sequence_tracker.align_with_gepa_candidates(candidate_data)
        
    #     # ✅ NOW GET CORRECT PER-CANDIDATE STATISTICS
    #     candidate_stats = sequence_tracker.get_candidate_summary()
        
    #     print(f"\n" + "="*80)
    #     print("📊 PER-CANDIDATE STATISTICS (CORRECTLY ALIGNED)")
    #     print("="*80)
        
    #     print(f"{'Cand #':<8} {'Mean Score':<12} {'Perplexity':<15} {'Confidence':<15} {'#Evals':<8}")
    #     print("-" * 80)
        
    #     for cand_id in sorted(candidate_stats.keys()):
    #         stats = candidate_stats[cand_id]
    #         score = stats['mean_score'] if stats['mean_score'] else 0
    #         ppl = stats['mean_perplexity'] if stats['mean_perplexity'] else 0
    #         conf = stats['mean_confidence'] if stats['mean_confidence'] else 0
    #         count = stats['count']
            
    #         print(f"{cand_id:<8} {score:<12.4f} {ppl:<15.2f} {conf:<15.4f} {count:<8}")



    # # Save the GEPA results for analysis
    # # results_dir = './dspy_vllm_jee_results'
    # # os.makedirs(results_dir, exist_ok=True)
    
    # # # --- FIX APPLIED HERE: Robust serialization logic ---
    # # try:
    # #     # Extract the final optimized prompt instructions directly
    # #     optimized_instructions = optimized_program.predict.signature.instructions

    # #     print(f"Optimized Instructions:\n{optimized_instructions}\n")
        
    # #     simple_results = {
    # #         "best_prompt": optimized_instructions,
    # #         "baseline_score": baseline_score.score,
    # #         "optimized_score": optimized_score.score,
    # #         "optimization_failed": False
    # #     }
        
    # #     # Save to a simplified file
    # #     filename = os.path.join(results_dir, "dspy_vllm_gepa_run_simplified.json")
    # #     with open(filename, 'w') as f:
    # #         json.dump(simple_results, f, indent=2)
        
    # #     print(f"Detailed GEPA optimization results saved to: {filename}")
        
    # # except Exception as e:
    # #     print(f"CRITICAL WARNING: Could not save full detailed results due to DSPy serialization bug: {e}")
    # #     print("Scores have been printed above, but the detailed prompt artifacts are NOT saved.")
    
    # # # Check for results and access the attribute
    # # if hasattr(optimized_program, 'detailed_results'):
    # #     results = optimized_program.detailed_results
        
    # #     candidate_data = []
        
    # #     # Zip candidates (program versions) with their aggregate scores
    # #     for i, (candidate, score) in enumerate(zip(results.candidates, results.val_aggregate_scores)):
    # #         # Extract the prompt from the main predictor in the candidate program
    # #         for name, predictor in candidate.named_predictors():
    # #             # For a ChainOfThought, the prompt is stored in the signature.instructions
    # #             prompt = predictor.signature.instructions
                
    # #             candidate_entry = {
    # #                 'candidate_idx': i,
    # #                 'score': score,
    # #                 'prompt': prompt,
    # #                 'prompt_length_chars': len(prompt),
    # #                 'prompt_length_words': len(prompt.split()),
    # #                 'predictor_name': name
    # #             }
                
    # #             candidate_data.append(candidate_entry)
    # #             break  # Stop after processing the first predictor in the module
                
    # #     print(f"Extracted {len(candidate_data)} candidates.")
    # #     print(candidate_data)

    # # else:
    # #     print("Error: detailed_results not found. Did you run GEPA with track_stats=True?")
