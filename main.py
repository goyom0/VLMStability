import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader 
from collections import defaultdict
import random
import multiprocessing as mp
import bitsandbytes as bnb
from functools import partial
from transformers import AutoTokenizer, AutoProcessor, pipeline, AutoModelForCausalLM, AutoModel, AutoModelForImageTextToText
import wandb
import traceback
from huggingface_hub import login
login(token=os.environ.get("HF_API_KEY"))
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
from accelerate import Accelerator



# 이미지 샘플러
class ImageBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        groups = defaultdict(list)
        for idx in range(len(dataset)):
            image_field = dataset.df.iloc[idx]["image"]
            img_key = dataset.get_image_key(image_field)
            groups[img_key].append(idx)
        self.batches = []
        for idxs in groups.values():
            for i in range(0, len(idxs), batch_size):
                self.batches.append(idxs[i:i + batch_size])
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
    def __len__(self):
        return len(self.batches)


def run_single_model(dataset_name, train_df, test_df, model_name, outdir, args):
    # ccelerator 초기화
    # gradient_accumulation_steps로 OOM 방지
    accelerator = Accelerator(gradient_accumulation_steps=8)
    device = accelerator.device

    from Evaluator import StabilityEvaluator, StabilityDataset, stability_collate_fn

    evaluator = None
    try:
        # Evaluator 초기화
        evaluator = StabilityEvaluator(
            dataset_name, train_df, test_df, model_name, 'open_source',
            gpt_key=os.getenv("OPENAI_API_KEY"),
            gemini_key=os.getenv("GEMINI_API_KEY"),
            device=device
        )

        if accelerator.is_main_process:
            print(f">>> Distributed Training on {accelerator.num_processes} GPUs")
            print(f"Weight check (Baseline): {sum(p.sum().item() for p in evaluator.model.parameters())}")

        if args.do_train == "train":
            train_set = StabilityDataset(train_df, evaluator.processor, evaluator.model)
            custom_collate = partial(stability_collate_fn, processor=evaluator.processor)
            
            sampler = ImageBatchSampler(train_set, args.batch_size)
            train_loader = DataLoader(
                train_set,
                batch_sampler=sampler,
                collate_fn=custom_collate,
                num_workers=0
            )
            
            save_path = os.path.join(outdir, f"checkpoints/{model_name.split('/')[-1]}")
            optimizer = bnb.optim.AdamW8bit(evaluator.model.parameters(), lr=args.lr)

            # _train 내부에서 accelerator.prepare(), EMA 업데이트 수행
            evaluator._train(
                model=evaluator.model,
                train_loader=train_loader,
                optimizer=optimizer,
                save_dir=save_path,
                loss_mode=args.loss_mode,
                lambda_kl=args.lambda_kl,
                temp=args.temp,
                kl_mode=args.kl_mode,
                estimator_type=args.estimator_type
            )

        # 평가
        if accelerator.is_main_process:
            print(f"--- [EVAL] Running evaluation with TRAINED model: {model_name} ---")
            p = 'visual'
            result_df = evaluator.run_evaluation(perturbation_type=p)
            if len(result_df) > 0:
                safe_model_name = model_name.split('/')[-1]
                suffix = f"_{args.loss_mode}" if args.do_train == "train" else "_baseline"
                out_file = f"{outdir}/{safe_model_name}_{p}{suffix}_results.csv"
                result_df.to_csv(out_file, index=False)
                print(f"Saved results: {out_file}")

    except Exception as e:
        if accelerator.is_main_process:
            print(f"!!! Error: {e}")
            traceback.print_exc()
    finally:
        if evaluator is not None:
            del evaluator
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--train_sample", type=int, default=None)
    ap.add_argument("--do_train", type=str, default="train", choices=["train", "saved_weight"])
    ap.add_argument("--loss_mode", type=str, default="combined")
    ap.add_argument("--lambda_kl", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--kl_mode", type=str, default="token_kl", choices=["token_kl", "token_jsd"])
    ap.add_argument("--estimator_type", type=str, default="k1", choices=["full_kl", "k3"])

    args = ap.parse_args()
    outdir = os.path.join(args.outdir, args.dataset)
    os.makedirs(outdir, exist_ok=True)
    
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # COCO
    train_df = pd.read_csv("/data/user/dahyoun/VLM/data/coco_dataset_new_2.tsv", sep='\t', on_bad_lines='skip')
    # NaturalBenchDataset
    test_df = pd.read_csv(f"/data/user/dahyoun/VLM/data/{args.dataset}_w_sensitivity.tsv", sep='\t', on_bad_lines='skip')

    if args.train_sample is not None:
        sampled_images = train_df["image"].drop_duplicates().sample(n=args.train_sample, random_state=42)
        train_df = train_df[train_df["image"].isin(sampled_images)].reset_index(drop=True)
        test_df = test_df.sample(n=min(int(args.train_sample*0.2), len(test_df)), random_state=42)

    run_single_model(args.dataset, train_df, test_df, model_name, outdir, args)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()


# CUDA_VISIBLE_DEVICES=1 python instruct_main_seq.py --outdir /data/user/dahyoun/VLM/results/rollout/token_kl_lam1 --dataset NaturalBenchDataset --train_sample 4000 --do_train train  --loss_mode combined --lambda_kl 1

