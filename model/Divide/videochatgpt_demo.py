import os
import argparse
import json
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from tqdm import tqdm
import torch
from PIL import Image
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=True)

    return parser.parse_args()

def preprocess_video_frames(video_frames, image_processor):
    preprocessed_frames = []
    for frame in video_frames:
        # Preprocess each frame
        preprocessed_frame = image_processor.preprocess(frame, return_tensors='pt')['pixel_values']
        # Clip the values to the range [0, 1]
        preprocessed_frame = torch.clamp(preprocessed_frame, 0, 1)
        preprocessed_frames.append(preprocessed_frame)
    
    # Combine all preprocessed frames into a single tensor
    preprocessed_video_tensor = torch.cat(preprocessed_frames, dim=0)
    return preprocessed_video_tensor

def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    # Load the ground truth file
    with open(args.gt_file) as file:
        gt_contents = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_contents):
        video_name = sample['video_name']
        sample_set = sample
        question = sample['Q']

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if video_path is not None:  # Modified this line
            video_frames = load_video(video_path)
            # Debug
            # print(video_frames) ## list 
        # Convert the video frames to a tensor
        # preprocessed_video_tensor = preprocess_video_frames(video_frames, image_processor)
        
        # try:
        # Run inference on the video and add the output to the list
        output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
        sample_set['pred'] = output
        output_list.append(sample_set)
        # except Exception as e:
        #     print(f"Error processing video file '{video_name}': {e}")

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)