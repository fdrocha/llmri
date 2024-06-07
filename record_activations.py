#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict
import util

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Custom Model Class
class LlamaActivations(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.activations = defaultdict(list)

    def forward(self, **kvs):
        self.activations = defaultdict(list)

        def capturer(name, i):

            def capture_activations(module, input, output):
                # Assuming you want to capture activations of the last hidden state of each layer
                # Output is a tuple of length 1, we take the element inside the tuple
                self.activations[name].append(output[0])

            return capture_activations

        handles = []
        for i, layer in enumerate(self.model.layers):
            for name, module in (("attn", layer.self_attn), ("mlp",
                                                             layer.mlp)):
                handle = module.register_forward_hook(capturer(name, i))
                handles.append(handle)

        output = super().forward(**kvs)

        # Remove hooks after the forward pass to avoid memory issues
        for handle in handles:
            handle.remove()

        return output


def get_model():
    model = LlamaActivations.from_pretrained(model_id,
                                             config=config,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto")
    #                                             local_files_only=True)
    # Gets rid of an annoying message when calling generate
    model.generation_config.pad_token_id = model.config.eos_token_id
    return model


def get_activations(model, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1)

    mlp_acts = model.activations["mlp"]
    attn_acts = model.activations["attn"]
    # mlp_acts members have shape (ContextLen, EmbeddingDim), so we stack them
    # to get (NLayers, ContextLen, EmbeddingDim)
    mlp_acts = torch.stack(mlp_acts, dim=0)
    # attn_cats have shape (1, ContextLen, EmbeddingDim) so we *cat* them
    # to get (NLayers, ContextLen, EmbeddingDim)
    attn_acts = torch.cat(attn_acts, dim=0)
    # We now want to interleave these in alternating layers, starting with attn
    nlayers, cl, ed = mlp_acts.shape
    acts = torch.stack([attn_acts, mlp_acts],
                       dim=1).reshape(2 * nlayers, cl, ed)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return acts, output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DIRECTORY', type=str)
    args = parser.parse_args()

    input_fn = os.path.join(args.DIRECTORY, 'inputs.txt')
    outputs_fn = os.path.join(args.DIRECTORY, 'outputs.txt')
    output_fn_mask = os.path.join(args.DIRECTORY, 'activations%06d.pt')

    lines = open(input_fn).readlines()

    outf = open(outputs_fn, 'w')
    model = get_model()
    for i, line in tqdm(list(enumerate(lines))):
        input_text = line.strip()
        text = utile.decode_line(input_text)
        try:
            acts, output_text = get_activations(model, text)
            output_fn = output_fn_mask % i
            torch.save(acts, output_fn)
            del acts
        except Exception as e:
            output_text = "ERROR " + str(e)
            print(f"ERROR on line {i}: {e}")
        outf.write(util.encode_line(output_text) + '\n')
        outf.flush()


if __name__ == '__main__':
    main()
