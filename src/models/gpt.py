"""
Code based on andrej karpathy minGPT code
https://github.com/karpathy/minGPT/
"""


import math
import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..utils.general_utils import CfgNode as CN

import time
import os

import json
from .layers import Block 

# -----------------------------------------------------------------------------
 

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = 'gpt'
        C.model_name = None
        # architecture parameters
        C.n_layer=4
        C.n_query_head=4
        C.n_kv_head=4
        C.n_embd=512
        C.block_size=1024
        C.vocab_size=512
        C.rope = False
        C.pretrained = None
        # dropout hyperparameters --> decrease for smaller dataset
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.out_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.rope = config.rope
        # self.config = config

        modules = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        )
        if self.rope==False:
            modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # casual attention mask
        self.register_buffer('mask', 1 - torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1, config.block_size, config.block_size))
        
        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def load_pretrained(self, model_path):
        pretrained_state_dict = torch.load(os.path.join(model_path, "model.pt"))
        old_block_size = 64
        with open(os.path.join(model_path,'config.json'), 'r') as file:
            old_config = json.load(file)
            old_block_size = old_config['data']['block_size']
        # Initialize the current model state dict
        self_state_dict = self.state_dict()

        # Loop over the pretrained state dict and update the corresponding weights
        for name, param in pretrained_state_dict.items():
            if name in self_state_dict:
                # If it's the wpe layer and sizes are different, handle separately
                if name == 'transformer.wpe.weight' and param.size(0) != self_state_dict[name].size(0):
                    # Copy the weights for the first 64 neurons
                    self_state_dict[name][:old_block_size, :] = param[:old_block_size, :]
                elif name.startswith("transformer.h.") and name.endswith(".attn.bias") and param.size()[2] != self_state_dict[name].size()[2]:
                    self_state_dict[name][:,:,:old_block_size,:old_block_size] = param
                    # Remaining weights are already randomly initialized
                else:
                    # Copy the weights for layers other than wpe
                    self_state_dict[name].copy_(param)

        # Load the updated state dict into the model
        self.load_state_dict(self_state_dict)


    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, fused=True)
        # maybe fuse? ckpt_path = os.path.join(out_dir, f'{config.model.name}.pt')
        return optimizer

    def forward(self, input_ids, attention_mask=None, labels=None):
        device = input_ids.device
        b, t = input_ids.size()
        #print(t, b, self.block_size)
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        if self.rope==False:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        causal_mask = self.mask[..., :t, :t]

        if attention_mask is not None:
            # expand attention_mask from (b, t) to (b, 1, 1, t) 
            expanded_mask = attention_mask[:, None, None, :]  # shape (b, 1, 1, t)
            # broadcast expanded_mask against (b, heads, t, t)
            combined_mask = causal_mask.masked_fill(expanded_mask == 0, float('-inf'))
        else:
            combined_mask = causal_mask


        print("DEBUG - INPUT IDs:\n", input_ids[0])


        for block in self.transformer.h:
            x = block(x, mask=combined_mask)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().to(device)
            shift_labels = labels[..., 1:].contiguous().to(device)
            # print("INSTANTIATING LOSS!")
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # ASSUMED PAD TOKEN ID is 0 IMPORTANT!!!!!
            # print("CALCULATING LOSS!!")
            loss = loss_fn(shift_logits.transpose(1, 2), shift_labels)
            # print("ABOUT TO RETURN!")
            return loss, logits
            
        else:
            return logits



    @torch.no_grad()
    

    def sample(self, start_tokens=None, size=1, temperature=1.0, max_new_tokens=1024,
               device=None, verbose=True, bos_token_id=1, pad_token_id=0, eos_token_id=2):

        if device is None:
            device = next(self.parameters()).device

        # from scratch generation
        if start_tokens is None:
            start_tokens = [[bos_token_id]]
            replicate_start = True

        elif isinstance(start_tokens[0], int):
            # single partial sequence, replicate for `size` parallel samples
            start_tokens = [start_tokens]  # make it a list of lists
            replicate_start = True
        
        else: # multiple partial sequences
            replicate_start = False
            if len(start_tokens) != size:
                size = len(start_tokens)
                print(f"Overriding `size` to match len(start_tokens) = {size}")

        # convert to a single 2D tensor: [size, seq_len_of_prompt]
        # if replicate_start, repeat same sequence for each item in batch.
        if replicate_start:
            prompt_ids = torch.tensor(start_tokens[0], dtype=torch.long).unsqueeze(0)  # [1, prompt_len]
            prompt_ids = prompt_ids.repeat(size, 1)  # [size, prompt_len]
        else:
            # have multiple distinct partial sequences
            # pad them to the same length if needed
            max_len_prompt = max(len(seq) for seq in start_tokens)
            padded_seqs = []
            for seq in start_tokens:
                # pad on left
                seq = [pad_token_id]*(max_len_prompt - len(seq)) + seq 
                padded_seqs.append(seq)
            prompt_ids = torch.tensor(padded_seqs, dtype=torch.long)  # [size, max_len_prompt]

        prompt_ids = prompt_ids.to(device)

        # start sampling
        for _ in trange(max_new_tokens, disable=not verbose, desc="Sampling"):
            # forward pass: shape => [size, current_seq_len, vocab_size]
            logits = self.forward(prompt_ids)
            if isinstance(logits, tuple):
                logits = logits[0]  # handle the (loss, logits) or other returns

            # take only the last time step's logits: shape [size, vocab_size]
            next_logits = logits[:, -1, :] / temperature
            next_probs = F.softmax(next_logits, dim=-1)

            # sample from the probability distribution
            next_token_ids = torch.multinomial(next_probs, num_samples=1)  # [size, 1]

            # append next_token_ids to the prompt
            prompt_ids = torch.cat([prompt_ids, next_token_ids], dim=1)  # [size, old_len+1]

        return prompt_ids


    
