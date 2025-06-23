import os
import pandas as pd
import argparse
import asyncio
import nest_asyncio
import random
import statistics as stats
from pathlib import Path

from data_loading import Data, DATA_DIR, Sample
from modeling import select_model
from reasoner import select_reasoner, Reasoner
from scorers import ExactScorer, GPTScorer
from typing import List

from tqdm import tqdm

import gc
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Script to evaluate model on puzzle benchmark")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use",
        required=True,
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="ref_puzzles",
        help="Name of the folder in data/ to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--reasoner",
        type=str,
        default="standard",
        help="Reasoner to use",
    )
    parser.add_argument(
        "--attempt_puzzles",
        action="store_true",
        help="Set this flag to run new solution attempts",
    )
    parser.add_argument(
        "--score_puzzles",
        action="store_true",
        help="Set this flag to run gpt scoring",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip existing files",
    )
    return parser.parse_args()

def evaluate(model_name: str, reasoner_name: str, reasoner: Reasoner, folder_path: str, output_dir: str, attempt_puzzles: bool, score_puzzles: bool, skip_existing=False, df_saved=None):
    data = Data.load_with_puzzle_content(folder_path)
    sample = data.sample
    path_out = Path(output_dir) / folder_path / model_name / f"{reasoner_name}.jsonl"
    path_out2 = Path(output_dir) / sample.title / model_name / f"{reasoner_name}.jsonl"
    # print("Output path: ", path_out)
    # print("Alternate path: ", path_out2)
    
    def fetch_existing():
        if os.path.exists(path_out):
            output_data = Data.load(path_out)
        elif os.path.exists(path_out2):
            output_data = Data.load(path_out2)
        else:
            return None
        return output_data
    
    existing_data = fetch_existing()
    if attempt_puzzles:
        if skip_existing and existing_data is not None:
            print(f"Skipping existing file {path_out}")
            sample.raw_output = existing_data.sample.raw_output
        else:
            sample.raw_output = reasoner.run(sample, sample.puzzle_content)
            data.save(path_out)
    else:
        if existing_data is None:
            raise Exception(f"File {path_out} and {path_out2} is missing, aborting")
        sample.raw_output = existing_data.sample.raw_output
        print(f"Using existing file {path_out}")
        # print(f"{data.sample.raw_output}")

    # Scoring
    scorer = GPTScorer()
    exact_scorer = ExactScorer()
    if score_puzzles:
        if skip_existing and df_saved is not None and sample.title in df_saved["puzzle_title"].values and df_saved[df_saved["puzzle_title"] == sample.title]["explanation"].values[0] != "No explanation available, run with --score_puzzles to get it":
            print(f"Skipping existing file {path_out}, using saved score")
            score, explanation = df_saved[df_saved["puzzle_title"] == sample.title]["score_ratio"].values[0], df_saved[df_saved["puzzle_title"] == sample.title]["explanation"].values[0]
        else:
            score, explanation = scorer.run_with_explanation(sample, folder_path=folder_path)
        print(f"Ran with gpt scorer, got score {score}")
    else:
        score = exact_scorer.run(sample)
        explanation = "No explanation available, run with --score_puzzles to get it"
    row = {
        "puzzle_title": sample.title,
        "correct": exact_scorer.run(sample),
        "score_ratio": score,
        "modalities": ", ".join(sample.modality),
        "skills": ", ".join(sample.skills),
        "candidate_output": sample.raw_output,
        "explanation": explanation,
        "reasoning_steps": "\n".join(map(lambda step: step.explanation, sample.reasoning))
    }
    
    torch.cuda.empty_cache()
    gc.collect()

    return sample, score, row

def run_statistics(scores: List[float], samples: List[Sample], rows: List[dict], df: pd.DataFrame):
    print("Average reasoning score: ", stats.fmean(scores))
    print("Reasoning Standard Deviation: ", stats.stdev(scores))
    correct_scores = list(map(lambda x: x["correct"], rows))
    print("Average correctness score: ", stats.fmean(correct_scores))
    print("Correctness Standard Deviation: ", stats.stdev(correct_scores))

def main():
    args = parse_args()
    folders = []
    puzzle_path = DATA_DIR / args.folder
    for folder in puzzle_path.iterdir():
        if folder.is_dir():
            folders.append(folder)
    if not folders:
        folders.append(puzzle_path)
        
    print(folders)
    scores = []
    samples = []
    results = []

    nest_asyncio.apply()

    random.seed(45)
    
    graded_output = os.path.join(args.output_dir, "graded", args.folder)
    save_path = os.path.join(graded_output, f"stepwise_{args.folder}_{args.model}.csv")
    
    print(f"Saving results to {save_path}")
    
    # if save_path.exists(), read the csv
    if os.path.exists(save_path):
        df_saved = pd.read_csv(save_path)
        print(f"Loaded existing results from {save_path}")
    else:
        df_saved = None
    
    model = select_model(args.model)
    reasoner = select_reasoner(args.reasoner, model)
    
    # ctr = 0

    for puzzle in tqdm(folders):
        # if ctr == 20:
        #     break
        sample, score, row_result = evaluate(
            model_name=args.model,
            reasoner_name=args.reasoner,
            reasoner=reasoner,
            folder_path=puzzle,
            output_dir=args.output_dir,
            attempt_puzzles=args.attempt_puzzles,
            score_puzzles=args.score_puzzles,
            skip_existing=args.skip_existing,
            df_saved=df_saved
        )


        scores.append(score)
        samples.append(sample)
        results.append(row_result)
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
    

    run_statistics(scores, samples, results, df)

    current_task = asyncio.current_task()
    print(f"Currently running coroutine: {current_task.get_name()}")

if __name__ == "__main__":
    main()