# VLMStability

### Dataset
- train: COCO
- test: NaturalBenchDataset

### Model
- Qwen2.5-VL-3B-Instruct

### Use
"accelerate launch main.py --outdir /your/output/path --dataset NaturalBenchDataset --train_sample 4000 --do_train train  --loss_mode combined --lambda_kl 1"
