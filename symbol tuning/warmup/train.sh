PRE_SEQ_LEN=128
LR=1e-4
NUM_GPUS=3

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file /home/gaohongfu/ChatGLM1/ptuning/train_shuxuejianmo.json \
    --validation_file /home/gaohongfu/ChatGLM1/ptuning/test_shuxuejianmo.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column label \
    --overwrite_cache \
    --model_name_or_path /home/gaohongfu/ChatGLM3/GPT2 \
    --output_dir output_shuxuejianmo/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 16



