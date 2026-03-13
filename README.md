# Clustering-Based Balanced Sampling and Allocation with Data Parallelism for High-Performance Fine-Tuning

## Training
```
accelerate launch -m magicoder.train_grad \
  --model_key codellama/CodeLlama-7b-Python-hf \
  --use_flash_attention True \
  --max_training_seq_length 1214 \
  --datafile_paths /path/to/data \
  --output_dir /path/to/output \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 128 \
  --group_by_length False \
  --ddp_find_unused_parameters False \
  --logging_steps 1 \
  --log_level info \
  --optim adafactor \
  --max_grad_norm -1 \
  --warmup_steps 15 \
  --learning_rate 5e-5 \
  --lr_scheduler_type linear \
  --prune "close" \
  --ratio 100 \
  --badge_batch 16 \
  --badge_forward_chunk_mult 8 \
  --badge_cleanup_interval 0 \
  --dataloader_num_workers 8 \
  --tf32 True
```