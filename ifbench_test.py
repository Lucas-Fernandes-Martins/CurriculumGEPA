import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

import IFBench.evaluation_lib as evaluation_lib

inputs_path = "ifbench_data/IFBench_test.jsonl"
_OUTPUT_DIR = "ifbench_data"


def ifeval_score(example, completion):

    input = evaluation_lib.InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"])
    response = {example["prompt"] : completion}

    scores = []
    feedback = []
    for func, output_file_name in [
        (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
        (evaluation_lib.test_instruction_following_loose, "eval_results_loose"),
    ]:
        output = func(input, response)
        accuracy = output.follow_all_instructions
        scores.append(evaluation_lib.return_scores([output]))
        feedback.append(evaluation_lib.return_report([output]))
    
    score = list(scores[0]['instruction-level'])[0]
    feedback = "List of followed and not followed constraints: " + feedback[0]
    #print("Score:", score)
    #print("Feedback:", feedback)

    return score, feedback


def benchmark_ifeval(prompt_to_response_path):
    inputs = evaluation_lib.read_prompt_list(inputs_path)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
    prompt_to_response_path)
    # get instruction following results
    for func, output_file_name in [
        (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
        (evaluation_lib.test_instruction_following_loose, "eval_results_loose"),
    ]:
        logging.info("Generating %s...", output_file_name)
        outputs = []
        for inp in inputs:
            outputs.append(func(inp, prompt_to_response))
        follow_all_instructions = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all_instructions) / len(outputs)
        logging.info("Accuracy: %f", accuracy)

        output_file_name = os.path.join(
            _OUTPUT_DIR, output_file_name + ".jsonl"
        )
        evaluation_lib.write_outputs(output_file_name, outputs)
        logging.info("Generated: %s", output_file_name)

        # Prints instruction following accuracy report.
        print("=" * 64)
        print(f"{output_file_name} Accuracy Scores:")
        scores = evaluation_lib.return_scores(outputs)
        print("Report:")
        print(evaluation_lib.return_report(outputs))
        print(scores)

        return scores, 
        #print(outputs)

if __name__ == "__main__":
    # prompt_to_response_path = "ifbench_data/sample_output.jsonl"
    # benchmark_ifeval(prompt_to_response_path)
    example = {"key": 1000, "prompt": "Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.", "instruction_id_list": ["punctuation:no_comma", "detectable_format:number_highlighted_sections", "length_constraints:number_words"], "kwargs": [{}, {"num_highlights": 3}, {"relation": "at least", "num_words": 300}]}
    completion = {"prompt": "Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.", "response": "Raymond III was the Count of Tripoli from 1152 to 1187 and Prince of Galilee and Tiberias in the Kingdom of Jerusalem. He was born in 1140 to Raymond II of Tripoli and Hodierna of Jerusalem. His mother was the daughter of Baldwin II of Jerusalem. \n\n*Early Life and Succession*\n\nRaymond III was only a child when his father was murdered. His mother Hodierna was regent until Raymond came of age. In 1155 Raymond married Eschiva the daughter of Walter I of Beirut. They had three children: Raymond IV Bertrand and a daughter who married Guy of Lusignan. \n\n*Reign*\n\nRaymond III's reign was marked by a series of conflicts and alliances with the Muslim world. He was involved in the defense of the Kingdom of Jerusalem against Nur ad-Din and later Saladin. He was also a key figure in the internal politics of the kingdom. He was a supporter of the queen mother Amalric of Jerusalem and opposed the succession of Guy of Lusignan. \n\n*Later Life and Death*\n\nIn 1187 Raymond III was part of the disastrous Battle of Hattin where the Christian forces were defeated by Saladin. He was one of the few to escape the battlefield but died later that year. His son Raymond IV succeeded him as Count of Tripoli. \n\nRaymond III's life and reign were marked by the complex politics and warfare of the Crusader states. He was a key figure in the defense of the Kingdom of Jerusalem and his death marked a significant loss for the Christian forces in the region. His legacy is a testament to the turbulent times in which he lived and the challenges faced by the Crusader states in their relations with the Muslim world."}
