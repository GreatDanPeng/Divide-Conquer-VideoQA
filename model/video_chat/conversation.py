from PIL import Image

import torch
import math
from transformers import StoppingCriteria, StoppingCriteriaList

from enum import auto, Enum

import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from models.video_transformers import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class Chat:
    def __init__(self, model, device='cuda:0'):
        self.device = device
        self.model = model
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        conv.messages.append([conv.roles[0], text + '\n'])
        return conv

    def answer(self, conv,  img_list, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0):
        conv.messages.append([conv.roles[1], None])
        # print("conv: ", conv)
        embs = self.get_context_emb(conv, img_list)
        # print("embs: ", embs)
        print("===Generated Answer===")
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text
    
    def summarize(self, conv, text1, text2, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0):
        conv.messages.append([conv.roles[1], None])
        # print("conv: ", conv)
        embs = self.get_context_emb_summary(conv, text1, text2)
        # print("embs: ", embs)
        print("===Generated Answer===")
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text
    
    # def answer_segments(self, conv,  img_list, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
    #            repetition_penalty=1.0, length_penalty=1, temperature=1.0):
    #     conv.messages.append([conv.roles[1], None])
    #     print("conv: ", conv)
    #     output_text_list = []
    #     print("img_list_len: ", len(img_list))

    #     for imgs in img_list:
    #         print("imgs: ", imgs[0].shape)
    #         embs = self.get_context_emb(conv, imgs)
    #         print("embs: ", embs)
    #         outputs = self.model.llama_model.generate(
    #             inputs_embeds=embs,
    #             max_new_tokens=max_new_tokens,
    #             stopping_criteria=self.stopping_criteria,
    #             num_beams=num_beams,
    #             do_sample=True,
    #             min_length=min_length,
    #             top_p=top_p,
    #             repetition_penalty=repetition_penalty,
    #             length_penalty=length_penalty,
    #             temperature=temperature,
    #         )
    #         output_token = outputs[0]
    #         if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
    #                 output_token = output_token[1:]
    #         if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
    #                 output_token = output_token[1:]
    #         output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    #         output_text = output_text.split('###')[0]  # remove the stop sign '###'
    #         output_text = output_text.split('Assistant:')[-1].strip()
    #         output_text_list.append(output_text)

    #     return output_text_list
        
    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def get_index_segments(self, num_frames, seg_num_segments, seg_size, seg_nums, seg_index, overlap, last_index):
        seg_seg_size = float(seg_size - 1) / seg_num_segments
        start = int(seg_seg_size / 2)
        if (seg_index == seg_nums - 1):
            last_seg_size = num_frames - seg_size * (seg_nums - 1)
            seg_seg_size = float(last_seg_size - 1) / seg_num_segments
            start = int(seg_seg_size / 2)
        offsets = np.array([
            int(seg_size * seg_index) + start + int(np.round(seg_seg_size * idx)) for idx in range(seg_num_segments)
        ])
        if overlap == True:
            if last_index != []:
                offsets = np.insert(offsets, 0, last_index)
            last_index = offsets[-seg_num_segments // 5:]

        return offsets, last_index

    def load_video(self, video_path, num_segments=8, return_msg=False):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        frame_indices = self.get_index(num_frames, num_segments)
        
        duration = len(vr) // vr.get_avg_fps()
        index = np.linspace(0, len(vr)-1, num=int(duration))
        buffer = vr.get_batch(index).asnumpy()
        # transform
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        
        transform = T.Compose([
            GroupScale(int(224), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])

        images_group = list()
        for frame in buffer:
            img = Image.fromarray(frame)
            images_group.append(img)
        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs_224 = transform(images_group)
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return torch_imgs_224, msg
        else:
            return torch_imgs_224
    
    def load_video_segments(self, video_path, num_segments=8, return_msg=False, seg_index=0, seg_time=30, overlap=False, last_index=[]):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        # print("num_frames: ", num_frames)
        avg_fps = vr.get_avg_fps()
        # avg_fps = 30
        seg_size = avg_fps * seg_time
        # print("seg_size: ", seg_size)
        seg_nums = math.ceil(len(vr) / seg_size)
        # print("seg_nums: ", seg_nums)
        frame_indices, last_index = self.get_index_segments(num_frames, num_segments, seg_size, seg_nums, seg_index, overlap, last_index)

        # transform
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
            
        transform = T.Compose([
            GroupScale(int(224), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
        # print("frame_indices: ", frame_indices)
        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs_224 = transform(images_group)

        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."

        if return_msg:
            return torch_imgs_224, msg, last_index
        else:
            return torch_imgs_224, last_index

    def upload_video(self, image, conv, img_list, num_segments):
        if isinstance(image, str):  # is a image path
            vid_chat, msg = self.load_video(image, num_segments=num_segments, return_msg=True)
            TC, H, W = vid_chat.shape
            image = vid_chat.reshape(1, TC//3, 3, H, W).to(self.device)
            # print("image: ", len(image[0]))

        else:
            raise NotImplementedError
        # print("Input video shape:", vid_chat.shape)
        image_emb, _ = self.model.encode_img(image)
        # print("image_emb: ", len(image_emb[0]))
        img_list.append(image_emb)
        conv.messages.append([
            conv.roles[0], 
            f"<Video><VideoHere></Video> {msg}\n"
        ])
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg, img_list, conv
    
    def upload_video_segments(self, image, conv, img_list, num_segments, seg_index, seg_time, overlap=False, last_index=[]):
        if isinstance(image, str):  # is a image path
            vid_chat, msg, last_index = self.load_video_segments(image, num_segments=num_segments, return_msg=True, seg_index=seg_index, seg_time=seg_time, overlap=overlap, last_index=last_index)
            TC, H, W = vid_chat.shape
            image = vid_chat.reshape(1, TC//3, 3, H, W).to(self.device)
            # print("image: ", len(image[0]))
        else:
            raise NotImplementedError
        # print("Input video shape:", vid_chat.shape)
        image_emb, _ = self.model.encode_img(image)
        # print("image_emb: ", len(image_emb[0]))
        img_list.append(image_emb)
        conv.messages.append([
            conv.roles[0], 
            f"<Video><VideoHere></Video>\n"
        ])
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg, img_list, conv, last_index
    
    def upload_img(self, image, conv, img_list):
        img = image#Image.open(image)#.convert('RGB')
        transform = T.Compose(
            [
                T.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        img = transform(img).unsqueeze(0).unsqueeze(0).cuda()
        image_emb, _ = self.model.encode_img(img)
        img_list.append(image_emb)
        conv.messages.append([
            conv.roles[0],
            f"<Image><ImageHere></Image>\n"
        ])
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg,img_list, conv

    def get_context_emb(self, conv, img_list):
        prompt = get_prompt(conv)
        # print(prompt)
        if '<VideoHere>' in prompt:
            prompt_segs = prompt.split('<VideoHere>')
        else:
            prompt_segs = prompt.split('<ImageHere>')
        # print("prompt_segs: ", len(prompt_segs))
        # print("img_list: ", img_list[0].shape)
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of visual placeholders and videos."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    
    def get_context_emb_summary(self, conv, text1, text2):
        prompt = get_prompt(conv)
        # print(prompt)
        prompt_segs = prompt.split('<TextHere>')
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        text_embs = [self.model.llama_model.model.embed_tokens(self.model.llama_tokenizer(text, return_tensors="pt").to(self.device).input_ids) for text in [text1, text2]]
        mixed_embs = [seg_embs[0], text_embs[0], seg_embs[1], text_embs[1], seg_embs[2]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs