import os
from utils.config import Config
from models.videochat import VideoChat
from utils.easydict import EasyDict
import torch
import math
from conversation import Chat
import tqdm
from decord import VideoReader, cpu
import json


# Set environment variables
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========================================
#             Model Initialization
# ========================================
def init_model():
    print('Initializing VideoChat')
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)
    model = VideoChat(config=cfg.model)
    model = model.to(torch.device(cfg.device))
    model = model.eval()
    chat = Chat(model)
    print('Initialization Finished')
    return chat

chat = init_model()

# ========================================
#             LLM Summarization
# ========================================
def do_summary(text):
    result = ""
    from transformers import pipeline
    summarizer = pipeline("summarization")
    return summarizer(text)[0]['summary_text']

# ========================================
#             Conquer with Summary
# ========================================
def two_by_two_summary(summary_list):
    if len(summary_list) == 1:
        return summary_list[0]
    elif len(summary_list) % 2 == 0:
        merged = [summary_list[i] + summary_list[i+1] for i in range(0, len(summary_list), 2)]
        for i in tqdm(range(len(merged))):
            merged[i] = do_summary(merged[i])
        return two_by_two_summary(merged)
    else:
        merged = [summary_list[i] + summary_list[i+1] for i in range(0, len(summary_list)-1, 2)]
        merged.append(summary_list[-1])
        for i in tqdm(range(len(merged[:-1]))):
            merged[i] = do_summary(merged[i])
        return two_by_two_summary(merged)

# ========================================
#    Generate Outputs to be summarized
# ========================================

def generate_outputs(video_path, seg_nums, seg_time, overlap=False, num_segments=16):

    output_text_list = []
    last_index = []
    question = 'This is a segment of a long video. Please summarize it detailedly.'

    for i in range(seg_nums):
        print("=" * 50)
        print("Segment Index: ", i)    
        print("=" * 50)
        conv = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        img_list = []
        msg, img_list, conv, last_index = chat.upload_video_segments(video_path, conv, img_list, num_segments=num_segments, seg_index=i, seg_time=seg_time, overlap=overlap, last_index=last_index)
        # print('last_index: ', last_index)
        # print(my_question)
        conv = chat.ask(question, conv)
        output_text = chat.answer(conv, img_list, max_new_tokens=1000)
        output_text = output_text.replace('\n', '')
        # print(output_text)
        output_text_list.append(output_text)
        # print('=' * 50)

    if overlap == True:
        conv = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        img_list = []
        msg, img_list, conv, last_index = chat.upload_video_segments(video_path, conv, img_list, num_segments=16, seg_index=seg_nums-1, seg_time=seg_time)
        conv = chat.ask(question, conv)
        final_output_text = chat.answer(conv, img_list, max_new_tokens=1000)
        final_output_text = final_output_text.replace('\n', '')
        # print(final_output_text)
        output_text_list.append(final_output_text)

    # print('-' * 100)
    # print(output_text_list)
    return output_text_list
    # json_string = json.dumps(output_text_list)
    # video_name = video_path.split('/')[-1].split('.')[0]
    # with open(f"outputs/{seg_time}/{video_name}_outputs.json", "w") as f:
    #     json.dump(json_string, f)

# ========================================
#            Generate QA Results
# ========================================

# write a method to generate results to a list which can later convert to .json file with format {{"video_path": "example/yoga.mp4", "num_frames": num_frames, "fps": fps,"num_segments": num_segments, segment1: "segment1_text", segment2: "segment2_text", ..., segmentn: "segmentn_text"}} where the segment1 to segmentn comes from each element in output_text_list
def generate_results(video_path, num_frames, fps, output_text_list, seg_size, seg_time):
    results = {"video_path": video_path, "num_frames": num_frames, "fps": fps, "seg_size": seg_size, "seg_time": seg_time}
    for i in range(len(output_text_list)):
        results[f"segment{i+1}"] = output_text_list[i]
    video_name = video_path.split('/')[-1].split('.')[0]
    with open(f"outputs/{seg_time}_overlap/{video_name}_results.json", "w") as f:
        json.dump(results, f)

# ========================================
#                Main
# ========================================
import time
time_list = []
for it in range(4):
    seg_time = 15 * (it + 1)
    num_segments = 12 * (it + 1)
    overlap = False
    if (it == 1):
        continue
    print("=" * 50)
    print("seg_time: ", seg_time)
    print("num_segments: ", num_segments)
    print("overlap: ", overlap)
    print("=" * 50)
    folder_path = '/root/autodl-tmp/Ask-Anything/video_chat/videos/'
    # file_names = os.listdir(folder_path)
    file_name = '100.mp4'
    # file_names = [f for f in file_names if os.path.isfile(os.path.join(folder_path, f))]
    # for file_name in file_names:
    start_time = time.time()
    print('video: ', file_name)
    # print("=" * 50)
    video_path = folder_path + file_name
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr) 
    avg_fps = vr.get_avg_fps()
    seg_size = avg_fps * seg_time
    seg_nums = math.ceil(num_frames / seg_size)
    output_text_list = generate_outputs(video_path, seg_nums, seg_time, overlap=overlap, num_segments=num_segments)
    end_time = time.time()
    print(f"Time for {it+1}th iteration: {end_time - start_time}")
    time_list.append(end_time - start_time)
    # generate_results(video_path, num_frames, avg_fps, output_text_list, seg_size, seg_time)

for i in range(len(time_list)):
    print(f"Time for {i+1}th iteration: {time_list[i]}")
print("=======Done=======")

        
# ========================================
#             Prompt and Questions
# ========================================

# conv = EasyDict({
#     "system": "",
#     "roles": ("Human", "Assistant"),
#     "messages": [],
#     "sep": "###"
# })
# conv.messages.append([
#     conv.roles[0], 
#     f"Segment1: <Text><TextHere></Text>\nSegment2: <Text><TextHere></Text>"
# ])
# # text1 = output_text_list[0]
# # text2 = output_text_list[1]
# conv = chat.ask(question, conv)
# result = chat.summarize(conv, text1, text2, max_new_tokens=1000)
    
# print(result)