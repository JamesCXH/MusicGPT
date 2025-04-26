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

# ── in main.py – put this near the top, after imports ────────────────
# def _write_midi(tok, tokens, path):
#     """
#     tok      : a miditok tokenizer
#     tokens   : 1-D torch.Tensor | list[int]
#     path     : output .mid/.midi file
#     """
#     if isinstance(tokens, torch.Tensor):
#         tokens = tokens.tolist()
#     tok([tokens]).dump_midi(path)          # <- wrap in a list --> 2-D
def _write_midi(tok, tokens, path):
    """
    tok      : a miditok tokenizer
    tokens   : 1-D torch.Tensor | list[int]
    path     : where to write the .mid/.midi file
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()          #  <-- keep it 1-D
    tok(tokens).dump_midi(path)           #  <-- no extra [ ... ]

# --------------------------------------------------------------------


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
    C.tokenizer_path = None

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
    # if config.pipeline.train_token:
    #     print("Tokenizer training not implemented. Using default tokenizer.")
    #     config.pipeline.train_token = False

    # -------------------------------------------------------------------------
    #  TOKENIZER  -------------------------------------------------------------
    # -------------------------------------------------------------------------
    if config.pipeline.train_token:
        # 1) Build a tokenizer *configuration* (tweak these params as you wish)

        BEAT_RES = {
            (0, 1): 8,
            (1, 2): 8,
            (1, 3): 3,
            (2, 4): 4,
            (2, 6): 3,
            (4, 8): 2,
        }

        TOKENIZER_PARAMS = {
            "pitch_range": (21, 108),  # exact 88-key range
            "beat_res": BEAT_RES,
            "num_velocities": 32,  # 4-point resolution
            "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
            "use_chords": False,  # simpler for solo piano
            "use_rests": True,
            "use_tempos": True,
            "use_time_signatures": True,
            "use_programs": False,
            "num_tempos": 32,
            "tempo_range": (40, 220),  # wider just in case
        }

        tokenizer = REMI(TokenizerConfig(**TOKENIZER_PARAMS))

        # 2) Train it on every MIDI file found under `config.data`
        midi_files = list(Path(config.data).glob("**/*.mid")) + \
                     list(Path(config.data).glob("**/*.midi"))

        # base vocab ≈ 400 → aim for 768 merges
        tokenizer.train(vocab_size=768, files_paths=midi_files)


        # 3) Save once so later runs can reuse it
        tok_path = Path(config.system.work_dir) / "768SmallerMidiTokenizer.json"
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tok_path)
        print(f"✅  Tokenizer saved to {tok_path}")
        #
        # # 4) Store the path in the runtime config (helpful if you later `wandb` log) from tokenizers
        # config.tokenizer_path = str(tok_path)
        # tokenizer_config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
        # tokenizer = REMI(tokenizer_config)
        # tok_path = Path(config.system.work_dir) / "DEFAULTMidiTokenizer.json"
        # tok_path.parent.mkdir(parents=True, exist_ok=True)
        # tokenizer.save(tok_path)
        # print(f"✅  Tokenizer saved to {tok_path}")

        # 4) Store the path in the runtime config (helpful if you later `wandb` log) from tokenizers
        config.tokenizer_path = str(tok_path)

    else:
        print("USING EXISTING TOKENIZER!!!")
        if config.tokenizer_path is None:
            print("❌  Pass --tokenizer_path=<path/to/MidiTokenizer.json> "
                  "or set pipeline.train_token=True to create one.")
            sys.exit(1)
        tokenizer = REMI(params=Path(config.tokenizer_path))
        print(f"✅  Loaded tokenizer from {config.tokenizer_path}")

    print(f"TOKENIZER LENGTH {len(tokenizer)}")

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
    if config.pipeline.sample:

        model.eval()

        sampled_tokens = model.sample(size=config.sample.n_scratch, max_new_tokens=config.model.block_size,
                                      device=None, verbose=True, bos_token_id=1, pad_token_id=0, eos_token_id=2)
        # DataCollator
        for i in range(config.sample.n_scratch):
            outmidi = os.path.join(out_dir, f"scratch{i+1}.midi")
            # tokenizer(sampled_tokens[i]).dump_midi(outmidi)
            _write_midi(tokenizer, sampled_tokens[i], outmidi)

        seed_sequences = []
        train_samples = []

        set_seed(None)

        for batch_idx, encodings in enumerate(dataloader):

            if batch_idx >= config.sample.n_seed:
                break

            input_ids = encodings["input_ids"]  # shape (B, T)

            # Pick a random sequence from the batch
            random_idx = np.random.randint(0, input_ids.size(0))

            # Get the tokens for that sequence as a list
            seed_sequence = input_ids[random_idx].tolist()
            seed_sequence = seed_sequence[:config.sample.seed_toks]

            train_samples.append(input_ids[random_idx])
            seed_sequences.append(seed_sequence)


        # Feed partial sequences as a prompts to the model
        generated_sequences = model.sample(start_tokens=seed_sequences, size=config.sample.n_seed,
                                 temperature=1.0, max_new_tokens=config.model.block_size-config.sample.seed_toks, device=None)


        # Save seeded samples
        for i in range(config.sample.n_seed):

            outmidi = os.path.join(out_dir, f"train_sample{i+1}.midi")
            # tokenizer(input_ids[random_idx]).dump_midi(outmidi)
            _write_midi(tokenizer, train_samples[i], outmidi)

            outmidi = os.path.join(out_dir, f"continued_sample{i+1}.midi")
            # tokenizer(generated_sequences[i]).dump_midi(outmidi)
            _write_midi(tokenizer, generated_sequences[i], outmidi)

        model.train()

    if config.pipeline.evaluate:
        pass
    

    