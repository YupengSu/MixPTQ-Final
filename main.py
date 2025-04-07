import time
import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import DEV, export_quant_table
from importlib.metadata import version

print('torch', version('torch'))
print('transformers', version('transformers'))
print('datasets', version('datasets'))
print('accelerate', version('accelerate'))
print('# of gpus', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights", seqlen=2048):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    model.seqlen = seqlen
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='llama model to load')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--method', type=str, default='gptq', choices=['gptq', 'rtn'], help='Quantization method to use.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--save', action='store_true', help='save model in pytorch format.')
    parser.add_argument('--save_safetensors', action='store_true', help='save model in safetensors format.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')

    parser.add_argument('--save_ppl', action='store_true', help='Save perplexity results.')
    parser.add_argument('--save_zeroshot', action='store_true', help='Save zero-shot results.')

    args = parser.parse_args()
    

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {model_name}")

    if 'llama' in model_name:
        from model.llama import quant_rtn, quant_gptq, llm_pack, load_quant, llm_multigpu, benchmark

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_llm(args.model)
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    if not args.load and args.wbits < 16:
        tick = time.time()
        if args.method == 'gptq':
            quantizers = quant_gptq(args, model, tokenizer, DEV)
        elif args.method == 'rtn':
            quantizers = quant_rtn(args, model, tokenizer, DEV)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llm_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.test_generation:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llm_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        from transformers import LlamaTokenizer, TextStreamer
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
        input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
        streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, streamer=streamer)

    RESULT_DIR = './results'
    RESULT_BASENAME = (
        f"{model_name.upper()}-W{args.wbits}A16G{args.groupsize}-{args.method.upper()}"
        if args.wbits < 16
        else f"{model_name.upper()}-BASELINE"
    )
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if args.save_ppl:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llm_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        from utils import eval_ppl
        ppl_test = eval_ppl(args, model, tokenizer, DEV)

        save_filepath = os.path.join(RESULT_DIR, f"PPL-{RESULT_BASENAME}.txt")
        with open(save_filepath, "w") as f:
            print(f"{'method':<15}{'actual_wbits':<15}{'wikitest2':<15}{'c4':<15}{'ptb':<15}", file=f, flush=True)
            print(f"{args.method:<15}{args.wbits:<15.4f}{ppl_test[0]:<15.4f}{ppl_test[1]:<15.4f}{ppl_test[2]:<15.4f}", file=f, flush=True)

    if args.save_zeroshot:
        save_fake = os.path.join(RESULT_DIR, f"FAKE-{RESULT_BASENAME}")
        if not os.path.exists(save_fake):
            os.makedirs(save_fake)

        model.save_pretrained(save_fake)
        tokenizer.save_pretrained(save_fake)

        from utils import eval_zeroshot
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zeroshot(args.model, save_fake, task_list, num_shot, accelerate)

        import shutil
        shutil.rmtree(save_fake) # delete the fake model

        print("********************************")
        print("zero_shot evaluation results")
        print(results)

        df = pd.DataFrame(results).T
        df_str = df.to_string()

        save_filepath = os.path.join(RESULT_DIR, f"ZEROSHOT-{RESULT_BASENAME}.txt")
        with open(save_filepath, "w") as f:
            print(df_str, file=f, flush=True)

    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if not args.observe and args.save:
        llm_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, f"{RESULT_BASENAME}.pt"))

    if not args.observe and args.save_safetensors:
        llm_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, os.path.join(RESULT_DIR, f"{RESULT_BASENAME}.safetensors"), metadata={"format": "pt"}, storage_type="disk")




