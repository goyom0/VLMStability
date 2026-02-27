

# Baseline: Cross-entropy
# Regularization: KL divergence or JSD
# loss: original image - visual perturbation
# same question (w/o textual perturbation)

import wandb
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
import base64
from io import BytesIO
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from openai import OpenAI
import re
from transformers import AutoProcessor, AutoModelForImageTextToText
from collections import Counter
import math
from qwen_vl_utils import process_vision_info
from scipy.special import kl_div
import torch.distributed as dist


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#########################
# Visual Perturbations
#########################

class PerturbationFunctions:
    @staticmethod
    def translation(image, n):
        img_np = np.array(image)
        # 이동 후 빈 공간을 검은색(0)으로 채움
        if n > 0: # Right
            res = np.pad(img_np, ((0, 0), (n, 0), (0, 0)), mode='constant')[:, :image.width, :]
        else: # Left
            res = np.pad(img_np, ((0, 0), (0, abs(n)), (0, 0)), mode='constant')[:, abs(n):, :]
        return Image.fromarray(res)

    @staticmethod
    def pad_crop(image, n):
        if n > 0: return TF.pad(image, padding=n, fill=0)
        return TF.center_crop(image, (image.height + n, image.width + n))

    @staticmethod
    def scale(image, factor=0.9, pad_fill=None):
        new_size = (int(image.height * factor), int(image.width * factor))
        scaled = TF.resize(image, new_size, interpolation=Image.BICUBIC)
        if pad_fill is None: return scaled
        pad_w = (image.width - scaled.width) // 2
        pad_h = (image.height - scaled.height) // 2
        return TF.pad(scaled, padding=(pad_w, pad_h, image.width - scaled.width - pad_w, image.height - scaled.height - pad_h), fill=pad_fill)

    @staticmethod
    def rotate(image, angle):
        return TF.rotate(image, angle, expand=True)

    @staticmethod
    def text_overlay(image, text):
        img_text = image.copy()
        draw = ImageDraw.Draw(img_text)
        draw.text((image.width // 4, image.height // 2), text, fill=(255, 0, 0))
        return img_text


def get_perturbation_map(img):
    pf = PerturbationFunctions
    p_map = {}

    # 1. Translation
    for n in range(-16, 17, 4):
        if n != 0: p_map[f"translation_{n}"] = pf.translation(img, n)

    # 2. Pad/Crop
    for n in range(-16, 17, 4):
        if n != 0: p_map[f"pad_crop_{n}"] = pf.pad_crop(img, n)

    # 3. Scale
    p_map["scale"] = pf.scale(img, 0.9) # no padding
    p_map["scale_black"] = pf.scale(img, 0.9, 0)
    p_map["scale_white"] = pf.scale(img, 0.9, 255)

    # 4. Rotation
    for angle in range(30, 360, 30):
        p_map[f"rotation_{angle}"] = pf.rotate(img, angle)

    # 5. Text Overlay
    texts = ["Answer_Yes", "Answer_No", "Maybe", "I_dont_know", "YES", "NO"]
    for t in texts:
        p_map[f"text_{t}"] = pf.text_overlay(img, t)

    return p_map


###################
# Training set 
###################
import hashlib

class StabilityDataset(Dataset):
    def __init__(self, df, processor, model, img_root=None):
        self.df = df
        self.processor = processor
        self.model = model
        self.img_root = img_root
        self.perturb_cache = {}

    def __len__(self):
        return len(self.df)

    # 이미지별 키 생성 
    def get_image_key(self, image_field):
        # case 1: image path
        if isinstance(image_field, str):
            if self.img_root is not None:
                return os.path.join(self.img_root, image_field)
            return image_field
        # case 2: base64 → hash
        return hashlib.md5(image_field.encode("utf-8")).hexdigest()

    def load_image(self, image_field):
        # 이미지 path인 경우
        if isinstance(image_field, str):
            if os.path.exists(image_field):
                return Image.open(image_field).convert("RGB")
        # Base64인 경우
        try:
            # Base64 패딩 보정
            missing_padding = len(image_field) % 4
            if missing_padding:
                image_field += '=' * (4 - missing_padding)
            return Image.open(
                BytesIO(base64.b64decode(image_field))
            ).convert("RGB")
        except Exception:
            # 경로도 없고 디코딩도 실패한 경우
            print(f"!!! [Image Error] Invalid path or Base64: {image_field[:50]}")
            return Image.new('RGB', (448, 448), (0, 0, 0)) # 혹은 스킵해도 되는데 일단 까만색 이미지로

    def _is_base64(self, s):
        if s.startswith('/') or s.startswith('./') or s.startswith('../'):
            return False
        if any(ext in s.lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            return False
        # 그 외에 길이가 너무 길면 Base64로 간주
        return len(s) > 200

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_field = row["image"]
        
        # 모든 이미지를 448x448로 강제 리사이징 (Qwen2.5-VL은 28x28 패치를 사용하기 때문!)
        fixed_res = (448, 448) 
        img_clean = self.load_image(image_field).resize(fixed_res)
        
        question = row['question']
        answer = str(row['answer']).lower()

        # 1. CE training prompt (정답 포함)
        msg_clean = [
            {"role": "user", "content": [{"type": "image", "image": img_clean}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        text_clean = self.processor.apply_chat_template(msg_clean, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(msg_clean)
        inputs_clean = self.processor(
            text=[text_clean],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # print("clean pixel_values:", inputs_clean["pixel_values"].shape)
        # print("clean grid:", inputs_clean["image_grid_thw"].shape)

        # KL/JSD sampling prompt (정답 제외)
        msg_sampling = [{"role": "user", "content": [{"type": "image", "image": img_clean}, {"type": "text", "text": question}]}]
        text_sampling = self.processor.apply_chat_template(msg_sampling, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msg_sampling)
        inputs_sampling = self.processor(
            text=[text_sampling],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # 전체 perturbation 처리
        img_key = self.get_image_key(image_field)
        if img_key not in self.perturb_cache:
            self.perturb_cache[img_key] = get_perturbation_map(img_clean)

        p_map = self.perturb_cache[img_key]
        perturbed_batch = []
        
        for p_name, img_pert in p_map.items():
            # perturbed 이미지도 원본과 동일한 해상도로 리사이징해야 함
            # 이렇게 해야 logQ_yp 계산할 때 gen_p의 토큰 개수와 perturbation의 특징량 개수가 일치할 수 있음
            img_pert_fixed = img_pert.resize(fixed_res)
            
            msg_p = [{"role": "user", "content": [{"type": "image", "image": img_pert_fixed}, {"type": "text", "text": question}]}]
            text_p = self.processor.apply_chat_template(msg_p, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(msg_p)
            in_p = self.processor(
                text=[text_p],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            perturbed_batch.append({
                 "p_name": p_name,
                "inputs": in_p
            })

        return {
            "origin": inputs_clean,
            "origin_sampling": inputs_sampling,
            "perturbed_list": perturbed_batch,
            "true_answer": answer
        }



def stability_collate_fn(batch, processor):
    def merge_features(input_dicts):
        # text
        text_features = [
            {
                "input_ids": d["input_ids"].squeeze(0),
                "attention_mask": d["attention_mask"].squeeze(0),
            }
            for d in input_dicts
        ]
        collated = processor.tokenizer.pad(text_features, return_tensors="pt")

        # vision: pixel_values
        if "pixel_values" in input_dicts[0]:
            pv_list = []
            thw_list = []

            for d in input_dicts:
                pv = d["pixel_values"]
                thw = d.get("image_grid_thw", None)

                # pixel_values -> batch dim
                if pv.dim() == 2:          # [N, C]
                    pv = pv.unsqueeze(0)   # [1, N, C]
                elif pv.dim() == 3 and pv.size(0) != 1:
                    pass
                elif pv.dim() == 3 and pv.size(0) == 1:
                    pass
                else:
                    raise RuntimeError(f"unexpected pixel_values shape: {pv.shape}")

                pv_list.append(pv)

                # grid_thw -> batch dim + pv와 일치하도록 보정
                if thw is None:
                    # grid가 없으면 N으로 추정 (N=정사각)
                    if pv.dim() == 3:
                        N = pv.size(1)
                        s = int(N ** 0.5)
                        if s * s != N:
                            raise RuntimeError(f"cannot infer grid from N={N}")
                        thw = torch.tensor([1, s, s], dtype=torch.long)
                    else:
                        raise RuntimeError("image_grid_thw missing for 4D pixel_values")
                if thw.dim() == 1:
                    thw = thw.unsqueeze(0)  # [1,3]

                # pv가 [1,N,C]일 때 grid의 (H*W)=N 아니면 고침
                if pv.dim() == 3:
                    N = pv.size(1)
                    HW = int(thw[0, 1].item()) * int(thw[0, 2].item())
                    if HW != N:
                        s = int(N ** 0.5)
                        if s * s != N:
                            raise RuntimeError(f"grid_thw mismatch and cannot fix: N={N}, thw={thw}")
                        thw = torch.tensor([[1, s, s]], dtype=torch.long)

                thw_list.append(thw)

            collated["pixel_values"] = torch.cat(pv_list, dim=0) # batch concat
            collated["image_grid_thw"] = torch.cat(thw_list, dim=0) 

        return collated

    origin = merge_features([b["origin"] for b in batch])
    origin_sampling = merge_features([b["origin_sampling"] for b in batch])

    all_pert, p_names, perturb_owner_idx = [], [], []
    for i, b in enumerate(batch):
        for p in b["perturbed_list"]:
            all_pert.append(p["inputs"])
            p_names.append(p["p_name"])
            perturb_owner_idx.append(i)

    perturbed_all = merge_features(all_pert)

    labels = processor.tokenizer(
        [b["true_answer"] for b in batch],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids

    return {
        "origin": origin,
        "origin_sampling": origin_sampling,
        "perturbed_all": perturbed_all,
        "perturbed_list_raw": all_pert,
        "perturb_owner_idx": perturb_owner_idx,
        "p_names": p_names,
        "label_ids": labels,
    }



##############
### Training
##############

class EMATeacher:
    def __init__(self, model, accelerator, alpha=0.999):
        unwrapped_model = accelerator.unwrap_model(model)  # DDP 벗길것!!
        
        # Teacher model
        self.model = type(unwrapped_model)(unwrapped_model.config)
        self.model.load_state_dict(unwrapped_model.state_dict())
        self.model.to(accelerator.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.alpha = alpha

    def update(self, model, accelerator):
        student_model = accelerator.unwrap_model(model)
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), student_model.parameters()):
                ema_param.data.mul_(self.alpha).add_(param.data, alpha=1 - self.alpha)



class StabilityEvaluator():
    def __init__(self, dataset, tr_df, ts_df, model_name, model_type, 
                 gpt_key=None, gemini_key=None, device=None):
        """
        - Datasets: 
          - Train: COCO (open-ended answer)
          - Test: Natural Bench (Binary - yes/no, A/B)
        - Model: QWEN-2.5-VL-3B-Instruct
        """
        self.device = device if device else "cuda"
        self.dataset = dataset
        self.model_name = model_name
        self.model_type = model_type
        self.tr_df = tr_df
        self.samples_df = ts_df
        self.seed = 0

        self.client = None
        self.tokenizer = None
                
        self.gpt_key = gpt_key
        self.gemini_key = gemini_key

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # device_map={"": self.device}
        )
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
    
    def _merge_sub_batch(self, input_dicts):
        text_features = [
            {
                "input_ids": d["input_ids"].squeeze(0),
                "attention_mask": d["attention_mask"].squeeze(0),
            }
            for d in input_dicts
        ]
        collated = self.processor.tokenizer.pad(
            text_features,
            return_tensors="pt"
        )
        if "pixel_values" in input_dicts[0]:
            collated["pixel_values"] = torch.cat(
                [d["pixel_values"] for d in input_dicts],
                dim=0
            )
        if "image_grid_thw" in input_dicts[0]:
            collated["image_grid_thw"] = torch.cat(
                [d["image_grid_thw"] for d in input_dicts],
                dim=0
            )
        return collated

    
    def _mc_token_kl_ema_loss(
        self,
        teacher_model,
        # student_logits,   # CE에서 나온 logits 재사용할 경우
        student_model,
        origin_inputs,
        pert_inputs,
        labels_full,
        temp=1.0,
        estimator_type="full_kl",
        topk=None,
    ):
        """
        Token-level KL aligned exactly with CE mask.
        KL is applied only on answer tokens (same positions as CE).
        """

        device = origin_inputs["input_ids"].device

        # Teacher
        with torch.no_grad():
            t_outputs = teacher_model(**origin_inputs)
            t_logits = t_outputs.logits  # [B, T, V]

        # Student (same text, perturbed image only)
        s_inputs = {
            "input_ids": origin_inputs["input_ids"],
            "attention_mask": origin_inputs["attention_mask"],
            "pixel_values": pert_inputs["pixel_values"],
            "image_grid_thw": pert_inputs["image_grid_thw"],
        }

        if "image_grid_thw" in pert_inputs:
            s_inputs["image_grid_thw"] = pert_inputs["image_grid_thw"]

        # s_logits = student_logits
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_outputs = student_model(**s_inputs)   # labels 없이 forward
            s_logits = s_outputs.logits

        # ----- Next-token alignment (same as CE) -----
        t_logits = t_logits[:, :-1, :] / temp
        s_logits = s_logits[:, :-1, :] / temp

        # CE와 동일한 위치 마스크
        # labels_full shape: [B, T]
        labels_full = labels_full[:, :t_logits.size(1) + 1]
        kl_mask = (labels_full[:, 1:] != -100)

        if kl_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ----- Select only answer token positions -----
        t_selected = t_logits[kl_mask] 
        s_selected = s_logits[kl_mask]

        # Top-K 사용할 경우
        if topk is not None and topk < t_selected.size(-1):
            k = min(topk, t_selected.size(-1))
            _, topk_idx = torch.topk(t_selected, k=k, dim=-1)
            t_selected = torch.gather(t_selected, dim=-1, index=topk_idx)
            s_selected = torch.gather(s_selected, dim=-1, index=topk_idx)

        if estimator_type == "k3":
            log_p = F.log_softmax(t_selected, dim=-1)
            log_q = F.log_softmax(s_selected, dim=-1)
            log_ratio = log_q - log_p
            kl_loss = (torch.exp(log_ratio) - 1 - log_ratio).mean() * (temp ** 2)
        else:
            t_probs = F.softmax(t_selected, dim=-1)
            s_log_probs = F.log_softmax(s_selected, dim=-1)
            kl_loss = F.kl_div(
                s_log_probs,
                t_probs,
                reduction="batchmean"
            ) * (temp ** 2)

        return kl_loss



    def _train(self, model, train_loader, optimizer, save_dir,
               loss_mode='combined', lambda_kl=1.0, temp=1.0, K=1,
               kl_mode="seq_jsd_mc", estimator_type='full_kl', topk=None):
        
        from accelerate import Accelerator, DistributedDataParallelKwargs
        import math
        import traceback

        torch.cuda.empty_cache()

        # find_unused_parameters=True 설정 추가해 frozen 레이어 에러 방지
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        final_save_dir = os.path.join(save_dir, kl_mode)
        os.makedirs(final_save_dir, exist_ok=True)

        accumulation_steps = 4  # 4 or 8??

        # kwargs_handlers에 위 설정을 전달
        accelerator = Accelerator(
            gradient_accumulation_steps=accumulation_steps, 
            kwargs_handlers=[ddp_kwargs]
        )
        self.accelerator = accelerator 
        device = accelerator.device
        
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.config.use_cache = False
                 
        model.train()
        ema_teacher = EMATeacher(model, accelerator, alpha=0.997)
        for p in ema_teacher.model.parameters():
            p.data = p.data.to(torch.bfloat16)
        
        # 비전 고정
        if hasattr(unwrapped_model, "visual"):
            unwrapped_model.visual.requires_grad_(False)
        elif hasattr(unwrapped_model, "model") and hasattr(unwrapped_model.model, "visual"): 
            unwrapped_model.model.visual.requires_grad_(False)
        else:
            for name, param in unwrapped_model.named_parameters():
                if "visual" in name:
                    param.requires_grad = False

        sub_bs = 1
        global_step = 0
        auto_factor = 1.0
        sum_ce, sum_kl = 0.0, 0.0
        cur_kl_val = 0.0
        final_loss = 0.0
        calibration_steps = 20
        count = 0

        # ===== wandb =====
        if accelerator.is_main_process:
            wandb.init(
                entity="goyom-yonsei-university",
                project="VLM",
                config={
                    "loss_mode": loss_mode,
                    "lambda_kl": lambda_kl,
                    "temp": temp,
                    "kl_mode": kl_mode,
                },
            )
            print(f"Start Training | Device: {device} | accumulation={accumulation_steps}, sub_bs={sub_bs}, kl_mode={kl_mode}")

        try:
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="train", disable=not accelerator.is_main_process)):
                # batch 준비
                origin = {k: v.to(device) for k, v in batch['origin'].items() if isinstance(v, torch.Tensor)}
                origin_sampling = {
                    k: v.to(device)
                    for k, v in batch['origin_sampling'].items()
                    if isinstance(v, torch.Tensor)
                }
                labels_batch = batch['label_ids'].to(device)
                input_ids = origin['input_ids']

                # CE loss (teacher forcing)
                labels_full = torch.full_like(input_ids, -100)
                label_pad_id = self.processor.tokenizer.pad_token_id
                if label_pad_id is None:
                    label_pad_id = -100
                for i in range(input_ids.size(0)):
                    target_tokens = labels_batch[i][labels_batch[i] != label_pad_id]
                    t_len = target_tokens.size(0)
                    if t_len == 0:
                        continue

                    seq_len = input_ids.size(1)
                    # answer가 input_ids 끝에 붙어있다고 가정하고, 실제 길이 초과 방지
                    t_len = min(t_len, seq_len)
                    labels_full[i, seq_len - t_len:seq_len] = target_tokens[-t_len:]
                
                if batch_idx % 1000 == 0 and accelerator.is_main_process:
                    valid_idx = (labels_full[0] != -100).nonzero(as_tuple=True)[0]
                    if len(valid_idx) > 0:
                        print(f"Target Token IDs: {labels_full[0][valid_idx]}")
                        print(f"Input IDs at same pos: {input_ids[0][valid_idx]}")

                with accelerator.accumulate(model):
                    # 그래프 초기화
                    total_kl_loss = torch.tensor(0.0, device=device)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs_clean = model(**origin, labels=labels_full)
                        student_logits = outputs_clean.logits
                        loss_ce = outputs_clean.loss
                        
                        cur_ce_val = loss_ce.item()

                    # EMA Teacher를 이용한 Anchor forward
                    with torch.no_grad():
                        ema_teacher.model.eval()
                        origin_s = {k: v for k, v in batch['origin_sampling'].items() if isinstance(v, torch.Tensor)}
                        print("pixel_values shape:", origin["pixel_values"].shape)
                        assert origin["pixel_values"].size(0) > 0
                        outputs_origin_teacher = ema_teacher.model(**origin_s)
                        model.train()

                    # ==============================
                    # perturbed loop + KL/JSD (Roll-out)
                    # ==============================
                    perturbed_raw_list = batch['perturbed_list_raw']
                    perturb_owner_idx = batch['perturb_owner_idx']
                    num_pert = len(perturbed_raw_list)
                    
                    # 각 GPU에서 계산된 KL 값 합산
                    local_kl_sum = 0.0 

                    if loss_mode in ['combined', 'kl_only']:
                        # 응답 시작 지점 찾기: labels_full에서 -100이 아닌 첫 번째 토큰 위치
                        # origin['input_ids'].size(1)을 쓰면 시퀀스 끝이라 슬라이싱 결과가 0이 됨
                        # labels_full이 -100인 구간(질문)이 끝나는 지점
                        valid_indices = (labels_full[0] != -100).nonzero(as_tuple=True)[0]
                        
                        if len(valid_indices) > 0:
                            # 정답 시작 위치
                            response_start_idx = valid_indices[0].item()
                        else:
                            # 만약 labels_full이 깨져있다면 뒤에서부터 고정 길이를 롤아웃으로 잡음
                            # Qwen2.5-VL의 일반적인 답변 길이를 고려하여 뒤에서 64토큰 강제 확보
                            response_start_idx = max(0, origin['input_ids'].size(1) - 64)

                        # 시작 인덱스가 시퀀스 끝과 같으면 KL은 무조건 0임. 최소 1토큰이라도 확보할 것
                        if response_start_idx >= origin['input_ids'].size(1) - 1:
                            response_start_idx = max(0, origin['input_ids'].size(1) - 2)

                        if count == calibration_steps:
                            auto_factor = min(sum_ce / (sum_kl + 1e-8), 1e6) 

                        if batch_idx == 0 and accelerator.is_main_process:
                            print(f"\nDEBUG: Response Start Index = {response_start_idx}")
                            print(f"DEBUG: Total Sequence Length = {origin['input_ids'].size(1)}")

                        for i in range(0, num_pert, sub_bs):
                            sub_list = perturbed_raw_list[i:i + sub_bs]
                            sub_pert_dict = self._merge_sub_batch(sub_list)
                            sub_pert = {k: v.to(device)
                                        for k, v in sub_pert_dict.items()
                                        if isinstance(v, torch.Tensor)}

                            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                # ---------- token_jsd (Roll-out) ----------
                                if kl_mode == "token_jsd":
                                    # origin_sampling forward (question only)
                                    out_origin = model(**origin_sampling)
                                    logits_origin = out_origin.logits[:, :-1, :] / temp

                                    # perturbed forward (question only)
                                    out_pert = model(**sub_pert)
                                    logits_pert = out_pert.logits[:, :-1, :] / temp

                                    # 공통 길이 정렬
                                    common_len = min(logits_origin.size(1), logits_pert.size(1))
                                    logits_origin = logits_origin[:, :common_len, :]
                                    logits_pert   = logits_pert[:, :common_len, :]

                                    # attention mask 기반 valid token 마스크
                                    mask_origin = origin_sampling["attention_mask"][:, 1:1+common_len]
                                    mask_pert = sub_pert["attention_mask"][:, 1:1+common_len]

                                    mask = (mask_origin & mask_pert).bool()

                                    if mask.sum() == 0:
                                        loss_kl = torch.tensor(0.0, device=device, requires_grad=True)
                                    else:
                                        P = F.softmax(logits_origin[mask], dim=-1)
                                        Q = F.softmax(logits_pert[mask], dim=-1)

                                        M = 0.5 * (P + Q)

                                        jsd = 0.5 * (
                                            (P * (torch.log(P + 1e-10) - torch.log(M + 1e-10))).sum(-1) +
                                            (Q * (torch.log(Q + 1e-10) - torch.log(M + 1e-10))).sum(-1)
                                        )

                                        loss_kl = jsd.mean()
                                
                                # ---------- token_kl (exact kl/k3 Roll-out) ----------
                                elif kl_mode == "token_kl":
                                    # response_start_idx를 사용하도록 전달
                                    loss_kl = self._mc_token_kl_ema_loss(
                                        ema_teacher.model,
                                        # student_logits,   # forward 재사용
                                        model,
                                        origin,
                                        sub_pert,
                                        labels_full,
                                        temp=temp,
                                        estimator_type=estimator_type,
                                        topk=256  # OOM 방지
                                    )

                                # loss_kl이 nan일 경우 방어
                                if not math.isnan(loss_kl.item()):
                                    kl_scale = num_pert / sub_bs
                                    total_kl_loss = total_kl_loss + (loss_kl / kl_scale)
                                    # total_kl_loss = total_kl_loss + (loss_kl.detach() / kl_scale)
                                    local_kl_sum += loss_kl.item() / kl_scale

                            del sub_pert

                        # 모든 GPU의 KL 값 평균 계산
                        kl_tensor = torch.tensor(local_kl_sum).to(device)
                        cur_kl_val = accelerator.reduce(kl_tensor, reduction='mean').item()

                    if loss_mode == "combined":
                        # CE 로스 + (스케일링 팩터 * KL loss)
                        final_loss = loss_ce + (lambda_kl * auto_factor * total_kl_loss)
                    elif loss_mode == "kl_only":
                        final_loss = lambda_kl * auto_factor * total_kl_loss
                    elif loss_mode == "baseline":
                        final_loss = loss_ce
                        auto_factor=1.0

                    accelerator.backward(final_loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 5.0)

                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if loss_mode is not 'baseline' and accelerator.sync_gradients:
                        ema_teacher.update(model, accelerator)
                    global_step += 1

                # calibration 및 로깅
                if accelerator.is_main_process:
                    if count < calibration_steps:
                        sum_ce += cur_ce_val
                        # sum_kl이 0이 되지 않도록 보정
                        sum_kl += max(abs(cur_kl_val), 1e-6) 
                        count += 1
                        if count == calibration_steps:
                            # factor 폭증 방지
                            raw_factor = sum_ce / (sum_kl + 1e-8)
                            auto_factor = min(raw_factor, 100000.0) 
                            print(f"\n>>> [CALIBRATION DONE] Factor Clamped: {raw_factor:.2f} -> {auto_factor:.2f}")

                    total_loss = cur_ce_val + auto_factor * cur_kl_val
                    wandb.log({"ce_loss": cur_ce_val, "kl_loss": cur_kl_val, 
                                "scaling_factor": auto_factor, "total_loss": total_loss}, 
                                step=global_step)        
                    if (batch_idx + 1) % accumulation_steps == 0:
                        print(f"\n[Step {batch_idx+1}] Loss: {total_loss:.4f} (CE: {cur_ce_val:.4f}, KL: {cur_kl_val:.4f}, factor: {auto_factor:.4f})")


                # 모든 GPU가 동일한 auto_factor를 갖도록 동기화
                if count >= calibration_steps:
                    factor_tensor = torch.tensor(auto_factor).to(device)
                    if dist.is_available() and dist.is_initialized():
                        dist.broadcast(factor_tensor, src=0)
                    auto_factor = factor_tensor.item()

                if (batch_idx + 1) % 1000 == 0 and accelerator.is_main_process:
                    self._save_checkpoint(unwrapped_model, save_dir)
                    torch.cuda.empty_cache()

        except Exception as e:
            if accelerator.is_main_process:
                print(f"Error!!! : {e}")
                traceback.print_exc()

        finally:
            if accelerator.is_main_process:
                wandb.finish()
                print(f"\n>>> Finalizing: Swapping Student with Teacher (EMA) for Inference...")
                
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.load_state_dict(ema_teacher.model.state_dict())

                self._save_checkpoint(unwrapped_model, final_save_dir)
                print(f">>> [DONE] Final Teacher model saved to: {final_save_dir}")
                torch.cuda.empty_cache()


    def _save_checkpoint(self, model, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print("\n[SAVE] Moving model to CPU for checkpointing...")
        model_cpu = model.to("cpu")
        torch.cuda.empty_cache()
        model_cpu.save_pretrained(
            save_dir,
            safe_serialization=False,
            max_shard_size="5GB" 
        )
        self.processor.save_pretrained(save_dir)
        model.to(self.device)
        torch.cuda.empty_cache()

        
    def get_prediction(self, model, image, question):
        import torch
        # 단답형 대답 지시 추가
        refined_question = f"{question}\nAnswer the question concisely with a single word or a short phrase."
        
        if self.model_type == 'open_source':
            # 1. Qwen 계열 (Qwen2.5-VL)
            if 'qwen' in self.model_name.lower():
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": refined_question}]}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)

            # 2. Phi-3.5-Vision
            elif 'phi' in self.model_name.lower():
                prompt = f"<|image_1|>\n<|user|>\n{refined_question}<|end|>\n<|assistant|>\n"
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)

            # 3. LLaVA
            else:
                prompt = f"USER: <image>\n{refined_question}\nASSISTANT:"
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)

            # generate
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=20, use_cache=False)
            
            # 입력 토큰 제외 후 디코딩
            input_token_len = inputs.input_ids.shape[1]
            generated_ids_trimmed = [out_ids[input_token_len:] for out_ids in generated_ids]

            final_output = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
            return final_output

        # 4. close_source
        else:
            if "gpt" in self.model_name.lower():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": refined_question},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                            ],
                        }
                    ],
                )
                return response.choices[0].message.content.strip()

            elif "gemini" in self.model_name.lower(): 
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[refined_question, image],
                    config={'temperature': 0.0, 'max_output_tokens': 16}
                )
                return response.text.strip()   

        return ""




    def run_evaluation(self, perturbation_type, limit=None):
        raw_logs = []
        df = self.samples_df.head(limit) if limit else self.samples_df

        self.model.eval()
        self.model.config.use_cache = True

        try:
            for i, row in tqdm(df.iterrows(), total=len(df), desc=self.model_name):
                try:

                    with torch.no_grad():
                        if not row['image'] or len(str(row['image'])) < 100: 
                            continue

                        img_data = base64.b64decode(row['image'])
                        img = Image.open(BytesIO(img_data)).convert('RGB')
                        question = row['question']
                        gt = str(row['answer']).lower()
                        
                        is_sensitive_label = row['rotation_label']

                        # 원본 예측 
                        orig_input = self._encode_image(img) if 'gpt' in self.model_name.lower() else img
                        orig_pred = self.get_prediction(self.model, orig_input, question)
                        orig_clean = self._clean_text(orig_pred)
                        orig_entropy, orig_dist, orig_avg_acc, orig_is_stable = self.calculate_entropy([gt], gt)

                        # 전체 답변 수집
                        all_preds_for_entropy = [orig_pred]
                        temp_results = [{
                            "p_name": "original", 
                            "pred": orig_pred, 
                            "is_stable": orig_is_stable 
                        }]

                        p_map = get_perturbation_map(img)
                        for p_name, p_img in p_map.items():
                            input_img = self._encode_image(p_img) if 'gpt' in self.model_name.lower() else p_img
                            pred = self.get_prediction(self.model, input_img, question)
                            all_preds_for_entropy.append(pred)
                            temp_results.append({
                                "p_name": p_name, 
                                "pred": pred, 
                                "is_stable": (self._clean_text(pred) == orig_clean)
                            })        
                        
                        entropy, dist, avg_acc, is_stable = self.calculate_entropy(all_preds_for_entropy, gt)

                        for res in temp_results:
                            raw_logs.append({
                                "image_id": row['index'],
                                "p_name": res['p_name'],
                                "prediction": res['pred'],
                                "ground_truth": gt,
                                "is_correct": (self._clean_text(res['pred']) == self._clean_text(gt)),
                                "is_stable": is_stable,
                                "is_rotation_sensitive": is_sensitive_label, 
                                "sample_entropy": entropy,
                                "sample_avg_acc": avg_acc,
                                "dist_log": str(dist)
                            })

                except Exception as e:
                    print(f"Error at index {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"Error at index {i}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            return pd.DataFrame(raw_logs)

    def _clean_text(self, text):
        if not text:
            return ""
        # 모든 종류의 공백 문자를 공백 하나로 치환 및 양끝 공백 제거
        text = " ".join(text.split()).lower()
        # 영숫자와 공백만 남기고 특수문자 제거
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        words = text.split()
        if not words: 
            return ""
        if 'yes' in words: return 'yes'
        if 'no' in words: return 'no'
        if 'maybe' in words: return 'maybe'
        if 'a' in words: return 'a'
        if 'b' in words: return 'b'
        return "not_found"

    def _encode_image(self, pil_img):
        # PIL 이미지를 Base64 문자열로 변환
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


    def calculate_entropy(self, predictions, ground_truth):
        if not predictions:
            return 0.0, {}, 0.0, 0

        n = len(predictions)
        preds_clean = [self._clean_text(p) for p in predictions]
        gt_clean = self._clean_text(str(ground_truth))
        
        counts = Counter(preds_clean)
        probs = [count / n for count in counts.values()]
        dist = {
            "unique_count": len(counts),
            "counts": dict(counts)
        }
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        is_stable = 1 if entropy < 1e-9 else 0
        accuracy = sum(1 for p in preds_clean if p == gt_clean) / n

        return round(entropy, 6), dist, round(accuracy, 4), is_stable
