import openai
from openai import OpenAI
import os
import argparse
import json
import ast
import csv
from collections import defaultdict
from multiprocessing.pool import Pool


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to the CSV file containing predictions.")
    parser.add_argument("--ground_truth_dir", required=True, help="The directory containing ground truth QA files.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save the combined annotation json file.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args


def load_predictions_from_csv(csv_path):
    """
    Load predictions from the CSV file and group them by experiment setting.
    """
    predictions_by_setting = defaultdict(dict)
    with open(csv_path, mode="r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            foldername = row["foldername"]
            video_id = row["filename"].split("_")[0]  # Extract video ID from filename
            segments = ast.literal_eval(row["segments"]) if row["segments"] else []
            summary_result = row["summary_result"]
            predictions_by_setting[foldername][video_id] = {
                "segments": segments,
                "summary_result": summary_result
            }
    return predictions_by_setting


def load_ground_truth(ground_truth_dir):
    """
    Load all ground truth QA pairs from the specified directory.
    """
    ground_truth_data = {}
    for file_name in os.listdir(ground_truth_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(ground_truth_dir, file_name)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    video_id = file_name.split(".")[0]
                    ground_truth_data[video_id] = data["global"]
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file: {file_name}")
                print(f"Error details: {e}")
    return ground_truth_data

def extract_answer(client, context, question):
    """
    Use GPT-3.5 to extract an answer to a question from a given context.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent assistant tasked with extracting answers to questions based on a given context. "
                        "If the context contains a clear answer to the question, extract it and return it in one sentence. "
                        "If the context does not contain enough information to answer the question, respond with 'N/A'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}\n\nFind out the answer.",
                },
            ],
        )
        # Extract content from the response
        print(response.choices[0].message.content)
        response_content = response.choices[0].message.content
        return response_content
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return "N/A"
    

def annotate(prediction_set, ground_truth_data, output_dir, experiment_name, experiment_scores):
    """
    Evaluates question and answer pairs using GPT-3.5 or GPT-4.
    """
    client = OpenAI()  # Initialize the OpenAI client
    for video_id, qa_pairs in ground_truth_data.items():
        if video_id not in prediction_set:
            print(f"Skipping {video_id} in experiment {experiment_name} as it has no prediction.")
            continue

        pred_data = prediction_set[video_id]
        predicted_segments = pred_data["segments"]
        predicted_summary = pred_data["summary_result"]

        for is_summary, predictions in [("segments", predicted_segments), ("summary", [predicted_summary])]:
            key = f"{experiment_name}_{is_summary}"  # Create key for experiment score tracking
            if key not in experiment_scores:
                experiment_scores[key] = {"total_score": 0, "count": 0}
            for idx, qa_pair in enumerate(qa_pairs):
                question = qa_pair["question"]
                correct_answer = qa_pair["answer"]

                # Combine predictions into a single context for extracting answers
                context = " ".join(predictions)

                # Extract the predicted answer using GPT-3.5
                pred_answer = extract_answer(client, context, question)

                try:
                    # Compute the correctness score
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content":
                                "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                                "- The predicted answer must capture the main themes and sentiments of the video.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Provide your evaluation of the contextual understanding of the prediction compared to the answer."                            
                            },
                            {
                                "role": "user",
                                "content":
                                    f"Please evaluate the following video-based question-answer pair:\n\n"
                                    f"Question: {question}\n"
                                    f"Correct Answer: {correct_answer}\n"
                                    f"Predicted Answer: {pred_answer}\n\n"
                                    "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                                    "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                    "For example, your response should look like this: {''score': 4}."
                            }
                        ]
                    )
                     # Extract content from the response
                    print(response.choices[0].message.content)
                    response_content = response.choices[0].message.content
                    response_dict = ast.literal_eval(response_content)  # Convert string to dictionary
                    score = response_dict["score"]
                    
                    # Update experiment score tracking
                    experiment_scores[key]["total_score"] += score
                    experiment_scores[key]["count"] += 1
                    
                    result_qa_pair = {
                        "question": question,
                        "correct_answer": correct_answer,
                        "predicted_answer": pred_answer,
                        "score": response_dict["score"],
                        "type": is_summary,
                    }
                    
                    # Save the question-answer pairs to a json file
                    setting_output_dir = os.path.join(output_dir, experiment_name)
                    os.makedirs(setting_output_dir, exist_ok=True)
                    output_file = os.path.join(setting_output_dir, f"{video_id}_{is_summary}_qa_{idx}.json")
                    with open(output_file, "w") as f:
                        json.dump(result_qa_pair, f)

                except Exception as e:
                    print(f"Error processing video '{video_id}', QA pair {idx}: {e}")
                    
def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    # Load predictions from CSV and group by experiment setting
    predictions_by_setting = load_predictions_from_csv(args.pred_path)

    # Load ground truth QA pairs
    ground_truth_data = load_ground_truth(args.ground_truth_dir)
    experiment_scores = {}
    
    # Generate output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    if not os.path.exists(args.output_json):
        with open(args.output_json, "w") as f:
            json.dump({}, f)

    # Run annotation for each experiment setting
    for experiment_name, prediction_set in predictions_by_setting.items():
        print(f"Running tests for experiment setting: {experiment_name}")
        annotate(prediction_set, ground_truth_data, args.output_dir, experiment_name, experiment_scores)

    # Combine all the processed files for each experiment setting into a single JSON
    combined_contents = {}
    for experiment_name in predictions_by_setting.keys():
        setting_output_dir = os.path.join(args.output_dir, experiment_name)
        if not os.path.exists(setting_output_dir):
            continue
        for file_name in os.listdir(setting_output_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(setting_output_dir, file_name)
                with open(file_path, "r") as f:
                    content = json.load(f)
                    combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(args.output_json, "w") as json_file:
        json.dump(combined_contents, json_file)
    
    
    # Compute and print final correctness scores for each experiment
    print("\nFinal CU Scores:")
    for experiment, scores in experiment_scores.items():
        average_score = scores["total_score"] / scores["count"] if scores["count"] > 0 else 0
        print(f"{experiment}: {average_score:.2f}")
    print("All evaluations completed!")


if __name__ == "__main__":
    main()
