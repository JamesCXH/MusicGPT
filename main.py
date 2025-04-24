import wandb
import os
import sys
import random
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator

import itertools

from src.models.gpt import GPT
from src.train.train import Trainer
from src.utils.general_utils import set_seed, setup_logging, save_train_log, CfgNode as CN
from src.utils.data_utils import get_data, split_data


wandb.login()

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out'

    # pipeline
    C.pipeline = CN()
    C.pipeline.train_token = False
    C.pipeline.train_gpt = True
    C.pipeline.evaluate = True
    C.pipeline.sample = True

    # data
    C.data = None

    # model
    C.model = GPT.get_default_config()

    # trainer
    C.gpt_trainer = Trainer.get_default_config()

    # sampling
    C.sample = CN()
    C.sample.n_scratch = 1
    C.sample.n_seed = 1
    C.sample.seed_toks = 512

    return C


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and override from the command line
    config = get_config()
    config.merge_from_args(sys.argv[1:])

    # check that train data is provided
    if config.data is None:
        print("No data path provided. Specify --data= argument.")
        sys.exit(1)

    set_seed(config.system.seed)

    if config.model.model_name is None:
        config.model.name = f'{config.model.model_type}_l{config.model.n_layer}_q{config.model.n_query_head}_kv{config.model.n_kv_head}'

    else:
        config.model.name = f'{config.model.model_type}_{config.model.model_name}_l{config.model.n_layer}_q{config.model.n_query_head}_kv{config.model.n_kv_head}'

    if config.model.rope : config.model.name += '_rope'

    # set up tokenizer 
    if config.pipeline.train_token:
        print("Tokenizer training not implemented. Using default tokenizer.")
        config.pipeline.train_token = False

    if not config.pipeline.train_token:
        tokenizer_config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
        tokenizer = REMI(tokenizer_config)

    # construct the model
    config.model.vocab_size = len(tokenizer)
    
    print("Run configuration:")
    print(config)

    model = GPT(config.model)
    
    if config.model.pretrained != None:
        print("Loading pretrained model...")
        pretrained_state_dict = torch.load(config.model.pretrained, weights_only=True)
        model.load_state_dict(pretrained_state_dict, strict=True)
    
    setup_logging(config)
    out_dir = config.system.work_dir

    wandb.init(project="MusicGen", config=config)

    dataloader = get_data(tokenizer, config.data, max_seq_len=config.model.block_size, batch_size=config.gpt_trainer.batch_size,
                              subsets=False, return_datasets=False, num_workers=config.gpt_trainer.num_workers)
    
    if config.pipeline.train_gpt:

        n_examples, train_loss = [], []

        # construct trainer object
        trainer = Trainer(config.gpt_trainer, model, dataloader)

        # iteration callback
        def batch_end_trainer_callback(trainer):
            
            wandb.log({"n_examples" : trainer.n_examples, "train_loss": trainer.loss})
            n_examples.append(trainer.n_examples)
            train_loss.append(trainer.loss.item())
            
            ckpt_path = os.path.join(out_dir, f'{config.model.name}.pt')

            if (trainer.n_iter + 1) % 1000 == 0:
                model.eval()
                torch.save(model.state_dict(), ckpt_path)
                print("SAVING A MODEL! (every 1000)")
            
                # revert model to training mode
                model.train()

        trainer.set_callback('on_batch_end', batch_end_trainer_callback)

        # run the optimization
        trainer.run()

        save_train_log(out_dir, n_examples, train_loss)

    # sample
    # if config.pipeline.sample:
    #
    #     sampled_tokens = model.sample(size=config.sample.n_scratch, max_new_tokens=config.model.block_size,
    #                                   device=None, verbose=True, bos_token_id=1, pad_token_id=0)
    #
    #     for i in range(config.sample.n_scratch):
    #         outmidi = os.path.join(out_dir, f"scratch{i+1}.mid")
    #         tokenizer(sampled_tokens[i]).dump_midi(outmidi)
    #
    #     seed_sequences = []
    #     train_samples = []
    #
    #     set_seed(None)
    #
    #     for batch_idx, encodings in enumerate(dataloader):
    #
    #         if batch_idx >= config.sample.n_seed:
    #             break
    #
    #         input_ids = encodings["input_ids"]  # shape (B, T)
    #
    #         # Pick a random sequence from the batch
    #         random_idx = np.random.randint(0, input_ids.size(0))
    #
    #         # Get the tokens for that sequence as a list
    #         seed_sequence = input_ids[random_idx].tolist()
    #         seed_sequence = seed_sequence[:config.sample.seed_toks]
    #
    #         train_samples.append(input_ids[random_idx])
    #         seed_sequences.append(seed_sequence)
    #
    #
    #     # Feed partial sequences as a prompts to the model
    #     generated_sequences = model.sample(start_tokens=seed_sequences, size=config.sample.n_seed,
    #                              temperature=1.0, max_new_tokens=config.model.block_size-config.sample.seed_toks, device=None)
    #
    #
    #     # Save seeded samples
    #     for i in range(config.sample.n_seed):
    #
    #         outmidi = os.path.join(out_dir, f"train_sample{i+1}.mid")
    #         tokenizer(input_ids[random_idx]).dump_midi(outmidi)
    #
    #         outmidi = os.path.join(out_dir, f"continued_sample{i+1}.mid")
    #         tokenizer(generated_sequences[i]).dump_midi(outmidi)

    # -------------------------------------------------------------------------
    #  SAMPLE -- generates scratch sequences *and* continuations of real data
    # -------------------------------------------------------------------------
    if config.pipeline.sample:

        # ------------------------------------------------------------------
        # special-token ids straight from the tokenizer
        # (never hard-code these numbers!)
        # ------------------------------------------------------------------
        pad_id = tokenizer['PAD_None']
        bos_id = tokenizer['BOS_None']
        eos_id = tokenizer['EOS_None']
        print(f"PAD ID {pad_id}")
        print(f"BOS ID {bos_id}")
        print(f"EOS ID {eos_id}")


        # ------------------------------------------------------------------
        # helpers
        # ------------------------------------------------------------------
        def save_midi(tok_tensor, fp):
            """Dump a tensor of ids to a .mid file."""
            tokenizer(tok_tensor.detach().cpu()).dump_midi(fp)


        def first_n_tokens(id_tensor, n=20):
            """
            Return the first n ids plus their readable token strings.
            Builds a reverse vocab dict only once.
            """
            if not hasattr(first_n_tokens, "_rev_vocab"):
                first_n_tokens._rev_vocab = {v: k for k, v in tokenizer.vocab.items()}
            ids = id_tensor[:n].tolist()
            toks = [first_n_tokens._rev_vocab.get(i, f"<unk:{i}>") for i in ids]
            return ids, toks


        # ------------------------------------------------------------------
        # SCRATCH (from-nothing) GENERATION
        # ------------------------------------------------------------------
        model.eval()  # disable dropout/LN noise
        with torch.no_grad():
            scratch_tokens = model.sample(
                size=config.sample.n_scratch,
                max_new_tokens=config.model.block_size,
                bos_token_id=bos_id,
                pad_token_id=pad_id,
                temperature=1.0,
            )

        # quick sanity check
        ids, toks = first_n_tokens(scratch_tokens[0], n=30)
        print("\n[SANITY] scratch ids :", ids)
        print("[SANITY] scratch toks:", toks)
        print("-" * 60)

        # save scratch midis
        for i, seq in enumerate(scratch_tokens):
            save_midi(seq, os.path.join(out_dir, f"scratch{i + 1}.mid"))

        # ------------------------------------------------------------------
        # SEED REAL TRAINING SEQUENCES & GENERATE CONTINUATIONS
        # ------------------------------------------------------------------
        seed_sequences, train_samples = [], []
        set_seed(None)  # true randomness for seed pick

        for batch in dataloader:
            if len(seed_sequences) >= config.sample.n_seed:
                break
            ids_batch = batch["input_ids"]
            rnd = np.random.randint(0, ids_batch.size(0))
            full_seq = ids_batch[rnd]
            seed = full_seq[:config.sample.seed_toks]
            train_samples.append(full_seq)
            seed_sequences.append(seed.tolist())

        with torch.no_grad():
            continued = model.sample(
                start_tokens=seed_sequences,
                size=len(seed_sequences),
                max_new_tokens=config.model.block_size - config.sample.seed_toks,
                bos_token_id=bos_id,
                pad_token_id=pad_id,
                temperature=1.0,
            )

        # sanity print for first pair
        seed_ids, seed_toks = first_n_tokens(torch.tensor(seed_sequences[0]), n=15)
        cont_ids, cont_toks = first_n_tokens(continued[0][len(seed_sequences[0]):], n=15)
        print("\n--- SEED (first 15) ------------------------")
        print(seed_ids)
        print(seed_toks)
        print("--- CONTINUATION (next 15) -----------------")
        print(cont_ids)
        print(cont_toks)
        print("-" * 60)

        # save seeds and continuations
        for i in range(len(seed_sequences)):
            save_midi(train_samples[i], os.path.join(out_dir, f"train_sample{i + 1}.mid"))
            save_midi(continued[i], os.path.join(out_dir, f"continued_sample{i + 1}.mid"))

        model.train()  # back to training mode
    # --------------------------- end pipeline.sample -----------------------------

    # evaluate
    if config.pipeline.evaluate:
        pass
    

    