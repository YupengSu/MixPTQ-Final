export HF_DATASETS_CACHE=/zju_0038/yupengsu/MixPTQ-Final/datasets/cache
export HF_DATASETS_TRUST_REMOTE_CODE=True
export http_proxy=http://localhost:20171
export https_proxy=http://localhost:20171

MODEL_NAME=llama3-8b    

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model /zju_0038/yupengsu/Models/${MODEL_NAME} --wbits 16 \
    --save_ppl --save_zeroshot \
    > logs/${MODEL_NAME}-baseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model /zju_0038/yupengsu/Models/${MODEL_NAME} --wbits 4 \
    --method rtn --true-sequential --act-order --groupsize 128 \
    --save  --save_ppl --save_zeroshot \
    > logs/${MODEL_NAME}-W4A16G128-RTN.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model /zju_0038/yupengsu/Models/${MODEL_NAME} --wbits 4 \
    --method gptq --true-sequential --act-order --groupsize 128 \
    --save --save_ppl --save_zeroshot \
    > logs/${MODEL_NAME}-W4A16G128-GPTQ.log 2>&1 &
