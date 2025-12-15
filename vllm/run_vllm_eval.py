import argparse
import json
import time
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# -----------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vLLM (optionally with EAGLE-3) on MT-Bench-style dataset with batched decoding."
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model name or path.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. Will load: data/{dataset}/question.jsonl",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Folder name to save outputs under {dataset}/",
    )

    parser.add_argument(
        "--use-eagle3",
        action="store_true",
        help="Enable EAGLE-3 speculative decoding.",
    )

    parser.add_argument(
        "--eagle3-draft",
        type=str,
        default=None,
        help="Draft model for EAGLE-3. Required if --use-eagle3 is set.",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="max_model_len for vLLM",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum generation tokens per turn.",
    )

    parser.add_argument(
        "--draft-depth",
        type=int,
        default=7,
        help="Drafting depth.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts per vLLM generate call.",
    )

    return parser.parse_args()


# -----------------------------------------------------------
# Load questions
# -----------------------------------------------------------
def load_questions(dataset):
    path = f"data/{dataset}/question.jsonl"
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qs.append(json.loads(line))
    return qs


# -----------------------------------------------------------
# Build vLLM model
# -----------------------------------------------------------
def build_vllm(args):
    speculative_config = None
    if args.use_eagle3:
        if args.eagle3_draft is None:
            raise ValueError("--use-eagle3 requires --eagle3-draft")
        speculative_config = {
            "method": "eagle3",
            "model": args.eagle3_draft,
            "num_speculative_tokens": args.draft_depth,
            "draft_tensor_parallel_size": 1,
        }
        print(f"[INFO] Using EAGLE-3 with draft: {args.eagle3_draft}")
    else:
        print("[INFO] Running base model only (no EAGLE-3).")

    llm = LLM(
        model=args.base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        speculative_config=speculative_config,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    return llm, tok


# -----------------------------------------------------------
# Main evaluation (batched, multi-turn aware)
# -----------------------------------------------------------
def run_eval(args):
    questions = load_questions(args.dataset)
    llm, tok = build_vllm(args)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Output directory
    out_dir = f"{args.dataset}/{args.experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    answer_file = os.path.join(out_dir, "answers.jsonl")
    stats_file = os.path.join(out_dir, "stats.json")

    total_tokens = 0
    total_time = 0.0
    num_turns = 0

    print(f"[INFO] Loaded {len(questions)} questions from data/{args.dataset}/question.jsonl")
    print(f"[INFO] Saving outputs to: {out_dir}")
    print(f"[INFO] Batch size: {args.batch_size}")

    answers_per_question = [[] for _ in questions]

    total_turns = sum(len(q.get("turns", [])) for q in questions)
    max_turns = max((len(q.get("turns", [])) for q in questions), default=0)

    pbar = tqdm(total=total_turns, desc="Decoding turns")

    for turn_idx in range(max_turns):
        active_indices = [
            i for i, q in enumerate(questions)
            if len(q.get("turns", [])) > turn_idx
        ]
        if not active_indices:
            continue

        for start_idx in range(0, len(active_indices), args.batch_size):
            batch_ids = active_indices[start_idx:start_idx + args.batch_size]

            prompts = []
            for qi in batch_ids:
                q = questions[qi]
                turns = q.get("turns", [])

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]

                for prev_t in range(turn_idx):
                    messages.append({"role": "user", "content": turns[prev_t]})
                    messages.append({
                        "role": "assistant",
                        "content": answers_per_question[qi][prev_t],
                    })

                messages.append({"role": "user", "content": turns[turn_idx]})

                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)

            start_time = time.time()
            outputs = llm.generate(
                prompts,
                sampling_params,
                use_tqdm=False,  # turn off vLLM process bar
            )
            elapsed = time.time() - start_time

            batch_tokens = 0
            for qi, out in zip(batch_ids, outputs):
                completion = out.outputs[0]
                text = completion.text
                out_tokens = len(completion.token_ids)

                batch_tokens += out_tokens
                answers_per_question[qi].append(text)

                num_turns += 1

            total_tokens += batch_tokens
            total_time += elapsed

            pbar.update(len(batch_ids))

    pbar.close()

    # write answers.jsonl
    with open(answer_file, "w", encoding="utf-8") as fout:
        for q, ans in zip(questions, answers_per_question):
            question_id = q.get("question_id", None)
            fout.write(json.dumps({
                "question_id": question_id,
                "answers": ans,
            }) + "\n")

    # Save statistics
    stats = {
        "dataset": args.dataset,
        "experiment_name": args.experiment_name,
        "base_model": args.base_model,
        "use_eagle3": args.use_eagle3,
        "eagle3_draft": args.eagle3_draft,
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "overall_throughput_tokens_per_s": (
            total_tokens / total_time if total_time > 0 else None
        ),
        "batch_size": args.batch_size,
    }

    with open(stats_file, "w", encoding="utf-8") as fs:
        json.dump(stats, fs, indent=2)

    print("\n======== SUMMARY ========")
    print(json.dumps(stats, indent=2))
    print("=========================\n")


# -----------------------------------------------------------
# Entry
# -----------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
