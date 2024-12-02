import openai
from openai import OpenAI
import os
import argparse
import json
import ast
import csv
from collections import defaultdict
from multiprocessing.pool import Pool


def load_predictions_from_folder(prediction_folder):
    """
    Load predictions from a folder containing multiple JSON files.
    """
    predictions = {}
    for file_name in os.listdir(prediction_folder):
        if file_name.endswith("_results.json"):
            file_path = os.path.join(prediction_folder, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                video_id = file_name.split("_")[0]  # Extract video ID from file name
                predictions[video_id] = data
    return predictions


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
    

def annotate(predictions, ground_truth_data, output_dir, experiment_scores):
    """
    Evaluates question and answer pairs using GPT-3.5 or GPT-4.
    """
    client = OpenAI() # Ensure OpenAI client is properly configured

    for video_id, ground_truth_qa_pairs in ground_truth_data.items():
        if video_id not in predictions:
            print(f"Skipping {video_id} as it has no prediction.")
            continue

        prediction = predictions[video_id]
        context = prediction.get("output_text_list", "")

        for idx, qa_pair in enumerate(ground_truth_qa_pairs):
            question = qa_pair["question"]
            correct_answer = qa_pair["answer"]

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
                print(response.choices[0].message.content)
                response_content = response.choices[0].message.content
                response_dict = ast.literal_eval(response_content)  # Convert string to dictionary
                score = response_dict["score"]

                # Update experiment score tracking
                if video_id not in experiment_scores:
                    experiment_scores[video_id] = {"total_score": 0, "count": 0}

                experiment_scores[video_id]["total_score"] += score
                experiment_scores[video_id]["count"] += 1

                result_qa_pair = {
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted_answer": pred_answer,
                    "score": score,
                }

                # Save the question-answer pair to a JSON file
                output_file = os.path.join(output_dir, f"{video_id}_qa_{idx}.json")
                with open(output_file, "w") as f:
                    json.dump(result_qa_pair, f)

            except Exception as e:
                print(f"Error processing video '{video_id}', QA pair {idx}: {e}")

                    
def main():
    """
    Main function to control the flow of the program.
    """
    score_name = "CU"
    # Define paths
    prediction_folder = "static/vanilla_outputs"  # Folder containing prediction JSON files
    ground_truth_dir = "static/MovieChat-1K_train/jsons"      # Directory containing ground truth QA JSON files
    output_dir = f"static/GPT4omini-assisted-Vanilla-Eval/Vanilla_{score_name}"            # Output directory for annotated results
    output_json = f"static/GPT4omini-assisted-Vanilla-Eval/Vanilla_{score_name}/combined_results.json"        # File for combined results

    # Load predictions from folder
    predictions = load_predictions_from_folder(prediction_folder)

    # Load ground truth QA pairs
    ground_truth_data = load_ground_truth(ground_truth_dir)

    # Initialize experiment scores
    experiment_scores = {}

    # Generate output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Annotate predictions
    annotate(predictions, ground_truth_data, output_dir, experiment_scores)

    # Combine all the processed files for each experiment setting into a single JSON
    combined_contents = {}
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as f:
                content = json.load(f)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a JSON file
    with open(output_json, "w") as json_file:
        json.dump(combined_contents, json_file)

    # Compute and print final correctness scores
    print(f"\nFinal {score_name} Scores:")
    scores = []
    for video_id, score in experiment_scores.items():
        average_score = score["total_score"] / score["count"] if score["count"] > 0 else 0
        scores.append(average_score)

    print(f"Average {score_name} Score: {sum(scores) / len(scores)}")

if __name__ == "__main__":
    main()