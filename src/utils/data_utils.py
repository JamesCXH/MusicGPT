import os
import sys
import json
import random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset
from miditok.pytorch_data import DatasetMIDI, DataCollator


def split_data(tokenizer, files_paths, subset_name, max_seq_len, split, augment):
    subset_chunks_dir = Path("..", f"Midi_{subset_name}")

    if split:
        # Split the MIDIs into chunks of sizes approximately about 1024 tokens
        split_files_for_training(files_paths=files_paths, 
                                tokenizer=tokenizer, 
                                save_dir=subset_chunks_dir, 
                                max_seq_len=max_seq_len, 
                                num_overlap_bars=2)

    if augment:
        # Perform data augmentation
        augment_dataset(subset_chunks_dir,
                        pitch_offsets=[-12, 12],
                        velocity_offsets=[-4, 4],
                        duration_offsets=[-0.5, 0.5])
    
    
def get_data(tokenizer, datapath, max_seq_len=1024, batch_size=64, subsets=True, return_datasets=False, split=True, augment=True, num_workers=12):
   
    # Search for both .mid and .midi extensions
    midipaths = list(Path(datapath).glob("**/*.mid")) + list(Path(datapath).glob("**/*.midi"))

    print("GETTING DATA!!!!")
    print("YEAH!!!!!")

    if not subsets:
        if not os.path.isdir((Path("..", "Midi_all"))): 
          split_data(tokenizer, midipaths, "all", max_seq_len, split, augment)
        
        midis = list(Path("..", "Midi_all").glob("**/*.mid")) + list(Path("..", "Midi_all").glob("**/*.midi")) 
        kwargs_dataset = {"max_seq_len": max_seq_len, "tokenizer": tokenizer, "bos_token_id": tokenizer["BOS_None"], "eos_token_id": tokenizer["EOS_None"]}
        dataset= DatasetMIDI(midis, **kwargs_dataset)
        print(f"Dataset size: {len(dataset)} sequences")
        
        if return_datasets:
            return dataset
        
        else:
            collator = DataCollator(pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, copy_inputs_as_labels=True)
            dataloader= DataLoader(dataset,
                                   batch_size=batch_size,
                                   collate_fn=collator,
                                   shuffle=True,
                                   num_workers=num_workers,  # AdamW
                                   pin_memory=True,  # good idea when using CUDA
                                   persistent_workers=True
                                   )
            print(f"Dataloader size: {len(dataloader)} batches")
            return dataloader
    

    
    else :  # Split MIDI paths in train/valid/test sets
        total_num_files = len(midipaths)
        num_files_valid = round(total_num_files * 0.15)
        num_files_test = round(total_num_files * 0.15)
        random.shuffle(midipaths)
        midi_paths_valid = midipaths[:num_files_valid]
        midi_paths_test = midipaths[num_files_valid:num_files_valid + num_files_test]
        midi_paths_train = midipaths[num_files_valid + num_files_test:]
    
        # Chunk MIDIs and perform data augmentation on each subset independently num_workers
        for files_paths, subset_name in ((midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")):
            split_data(tokenizer, files_paths, subset_name, max_seq_len, split, augment)

        # Create Dataset and Collator for training
        midi_paths_train = list(Path("..", "Midi_train").glob("**/*.mid")) + list(Path("..", "Midi_train").glob("**/*.midi"))
        midi_paths_valid = list(Path("..", "Midi_valid").glob("**/*.mid")) + list(Path("..", "Midi_valid").glob("**/*.midi"))
        midi_paths_test = list(Path("..", "Midi_test").glob("**/*.mid")) + list(Path("..", "Midi_test").glob("**/*.midi"))
        kwargs_dataset = {"max_seq_len": max_seq_len, "tokenizer": tokenizer, "bos_token_id": tokenizer["BOS_None"], "eos_token_id": tokenizer["EOS_None"]}
        dataset_train = DatasetMIDI(midi_paths_train, **kwargs_dataset)
        print(f"Train Dataset size: {len(dataset)} files")
        dataset_valid = DatasetMIDI(midi_paths_valid, **kwargs_dataset)
        print(f"Valid Dataset size: {len(dataset)} files")
        dataset_test = DatasetMIDI(midi_paths_test, **kwargs_dataset)
        print(f"Test Dataset size: {len(dataset)} files")
    
        if return_datasets:
            return {"train": dataset_train, "valid": dataset_valid, "test": dataset_test}
        
        else:
            collator = DataCollator(pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, copy_inputs_as_labels=True)
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collator)
            dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, collate_fn=collator)
            dataloader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collator)

            return {"train": dataloader_train, "valid": dataloader_valid, "test": dataloader_test}
    



