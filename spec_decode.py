import argparse
import subprocess
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-num", type=int, default=0, help="Device number to use for CUDA")
    parser.add_argument("--target-model", choices=["llama3-8b", "llama2-13b"], required=True)
    parser.add_argument("--method", choices=["eagle", "hass", "poss-1", "poss-2", "poss-3"], required=True)
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--total-token", type=int, default=60, help="Number of tokens to verify at each draft-verfication round")
    parser.add_argument("--depth", required=True, type=int, help="Number of positions to draft at each draft-verfication round")
    parser.add_argument("--repeat-time", type=int, default=1, help="Number of times to repeat the generation")
    parser.add_argument("--dataset", choices=["mt_bench", "alpaca", "gsm8k", "humaneval", "qa", "sum"], required=True)
    args = parser.parse_args()

    # Check if CUDA is available.
    device_num = check_cuda_availability(args.device_num)

    # Determine the generation code based on the target model and method.
    generation_code = determine_generation_code(args.target_model, args.method)

    # Determine the draft and target model.
    draft_path, target_path = determine_draft_and_target_model(args.target_model, args.method)

    # Determine the poss related settings.
    forward_num_total, position_per_layer = determine_poss_settings(args.target_model, args.method)

    # In the codebase, the depth starts from 0. So it minus 1 to align with the paper.
    args.depth = args.depth - 1

    # Generate the model ID.
    model_id = f"{args.target_model}-{args.method}-depth{args.depth+1}-tt{args.total_token}"
    for iter in range(args.repeat_time):
        model_id += f"-{iter}"
        cmd = [
            f"CUDA_VISIBLE_DEVICES={device_num}", "python", "-m", generation_code,
            "--ea-model-path", draft_path,
            "--base-model-path", target_path,
            "--temperature", f"{args.temperature}",
            "--model-id", model_id,
            "--forward_num_total", f"{forward_num_total}",
            "--position_per_layer", f"{position_per_layer}",
            "--bench-name", args.dataset,
            "--total-token", f"{args.total_token}",
            "--depth", f"{args.depth}",
        ]

        # Run the command as a shell command for the CUDA environment variable to work
        subprocess.run(" ".join(cmd), shell=True)

def check_cuda_availability(device_num):
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available on this machine. Using CPU instead.")
    num_devices = torch.cuda.device_count()
    if device_num < 0 or device_num >= num_devices:
        print(f"WARNING: Invalid device number {device_num}. Available devices: 0 to {num_devices - 1}. Using device 0 instead.")
        device_num = 0
    return device_num


def determine_generation_code(target_model, method):
    if target_model == "llama3-8b":
        if method in ["poss-1", "poss-2", "poss-3"]:
            return "evaluation.gen_poss_answer_llama3chat"
        else:
            return "evaluation.gen_ea_answer_llama3chat"
    elif target_model == "llama2-13b":
        if method in ["poss-1", "poss-2", "poss-3"]:
            return "evaluation.gen_poss_answer_llama2chat"
        else:
            return "evaluation.gen_ea_answer_llama2chat"
    else:
        raise ValueError("Unsupported target model specified.")

def determine_draft_and_target_model(target_model, method):
    if target_model == "llama3-8b":
        target_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        draft_path_map = {
            "eagle": "HINT-lab/EAGLE-Llama3-8B-Instruct-Reproduce",
            "hass": "HINT-lab/HASS-Llama3-8B-Instruct-Reproduce",
            "poss-1": "HINT-lab/PosS1-Llama3-8B-Instruct",
            "poss-2": "HINT-lab/PosS2-Llama3-8B-Instruct",
            "poss-3": "HINT-lab/PosS3-Llama3-8B-Instruct"
        }
        draft_path = draft_path_map[method]
    elif target_model == "llama2-13b":
        target_path = "meta-llama/Llama-2-13b-chat-hf"
        draft_path_map = {
            "eagle": "yuhuili/EAGLE-llama2-chat-13B",
            "hass": "HArmonizedSS/HASS-LLaMA2-Chat-13B",
            "poss-1": "HINT-lab/PosS1-Llama2-13B-Chat",
            "poss-2": "HINT-lab/PosS2-Llama2-13B-Chat",
            "poss-3": "HINT-lab/PosS3-Llama2-13B-Chat"
        }
        draft_path = draft_path_map[method]
    else:
        raise ValueError("Unsupported target model specified.")
    return draft_path, target_path

def determine_poss_settings(target_model, method):
    # The first return value is forward_num_total, which is the max depth at training time.
    # The second is the position per layer.
    # The total layers a model has is forward_num_total/position_per_layer
    if target_model == "llama3-8b":
        if method == "poss-1":
            return 7, 1
        elif method == "poss-2":
            return 8, 2
        elif method == "poss-3":
            return 12, 3
    elif target_model == "llama2-13b":
        if method == "poss-1":
            return 6, 1
        elif method == "poss-2":
            return 6, 2
        elif method == "poss-3":
            return 6, 3
    else:
        raise ValueError("Unsupported target model or method specified.")

if __name__ == "__main__":
    main()
