import os
import time
import json
from utils.config import Config
from models.videochat import VideoChat
from utils.easydict import EasyDict
import torch
from conversation import Chat

# Set environment variables
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========================================
#             Model Initialization
# ========================================
def init_model():
    print("Initializing VideoChat")
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)
    model = VideoChat(config=cfg.model)
    model = model.to(torch.device(cfg.device))
    model = model.eval()
    chat = Chat(model)
    print("Initialization Finished")
    return chat


chat = init_model()

# ========================================
#    Generate Outputs to be Summarized
# ========================================

def generate_outputs(video_path, ground_truth_file, num_frames=32):
    """
    Generate outputs by asking questions to the model based on a specific ground truth file.
    """
    output_text_list = []

    # Load ground truth questions
    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")

    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)
        questions = [qa["question"] for qa in ground_truth.get("global", [])]

    print("=" * 50)
    conv = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    img_list = []

    # Upload video and get initial context
    msg, img_list, conv = chat.upload_video(video_path, conv, img_list, num_segments=num_frames)

    # Ask each question and collect answers
    for question in questions:
        print(f"Question: {question}")
        conv = chat.ask(question, conv)
        output_text = chat.answer(conv, img_list, max_new_tokens=1000)
        output_text = output_text.replace("\n", "")
        output_text_list.append(output_text)

    return output_text_list


# ========================================
#            Generate QA Results
# ========================================

def generate_results(video_path, output_text_list):
    """
    Generate a structured results JSON file.
    """
    video_name = video_path.split("/")[-1].split(".")[0]
    num_frames = len(output_text_list)
    results = {
        "video_path": video_path,
        "num_frames": num_frames,
        **{f"segment{i+1}": text for i, text in enumerate(output_text_list)}
    }

    output_file = f"vanilla_outputs/{video_name}_results.json"
    os.makedirs("vanilla_outputs", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    return results


# ========================================
#                Main
# ========================================

def main():
    # Initialize variables
    time_list = []
    num_frames = 32

    # Print configuration
    print("=" * 50)
    print("Number of frames per video:", num_frames)
    print("=" * 50)

    # Folder paths
    folder_path = "/root/autodl-tmp/Ask-Anything/video_chat/videos/"
    ground_truth_dir = "/root/autodl-tmp/Ask-Anything/video_chat/jsons/"

    # Get the list of video files
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

    # Process each video
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        video_name = video_file.split(".")[0]
        ground_truth_file = os.path.join(ground_truth_dir, f"{video_name}.json")

        print(f"Processing video: {video_file}")

        try:
            # Start timing
            start_time = time.time()

            # Generate outputs for the video
            output_text_list = generate_outputs(video_path, ground_truth_file, num_frames=num_frames)
            print(f"Generated outputs: {output_text_list}")

            # Save the results
            generate_results(video_path, output_text_list)

            # End timing
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken for {video_file}: {elapsed_time:.2f} seconds")

            # Append to time list
            time_list.append({"video": video_file, "time": elapsed_time})

        except Exception as e:
            print(f"Error processing {video_file}: {e}")

    # Print summary
    print("\n======= Summary =======")
    for entry in time_list:
        print(f"Video: {entry['video']}, Time: {entry['time']:.2f} seconds")

    print("======= Done =======")


if __name__ == "__main__":
    main()