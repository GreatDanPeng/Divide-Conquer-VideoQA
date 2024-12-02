import os
import json
from collections import defaultdict


def calculate_average_scores(output_dir):
    """
    Calculate average correctness scores for each experiment type from JSON files.
    """
    # Dictionary to store total scores and counts for each experiment
    experiment_scores = defaultdict(lambda: {"total_score": 0, "count": 0})

    # Iterate over all subdirectories (experiment settings)
    for experiment_name in os.listdir(output_dir):
        experiment_path = os.path.join(output_dir, experiment_name)
        if not os.path.isdir(experiment_path):
            continue

        # Process all JSON files in the experiment directory
        for file_name in os.listdir(experiment_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(experiment_path, file_name)
                with open(file_path, "r") as f:
                    result = json.load(f)
                    score = result.get("score", 0)
                    qa_type = result.get("type", "unknown")

                    # Update total score and count for the experiment type
                    key = f"{experiment_name}_{qa_type}"
                    experiment_scores[key]["total_score"] += score
                    experiment_scores[key]["count"] += 1

    # Compute and print average scores for each experiment
    print("\nFinal DO Scores:")
    for experiment, scores in experiment_scores.items():
        average_score = scores["total_score"] / scores["count"] if scores["count"] > 0 else 0
        print(f"{experiment}: {average_score:.2f}")


if __name__ == "__main__":
    # Specify the output directory where JSON files are stored
    output_dir = "static/GPT3.5_DO"  # Replace with the path to your output directory
    calculate_average_scores(output_dir)