import argparse
import os
from pathlib import Path
import random
import time
import torch
import torch.optim
from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import AdamW, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup
from tqdm import tqdm

import wandb

from utils import GptData, format_time


def parse():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('-d', '--data', type=str, choices=['lotr', 'hp'],
                        default='lotr', help='dataset for training')
    parser.add_argument('--max_length', type=int, default=None, 
                        help='acceptable max length of a sentence')

    # hyperparams
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=4, help='epochs')

    # misc.
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--gpu', action='store_false', default=True)
    parser.add_argument('--id', type=str, default=f"{time.strftime('%d-%m-%H%M', time.localtime())}")
    
    args = parser.parse_args()

    return args


def main():
    args = parse()
    config = vars(args)

    wandb.init(project='text-gen', entity='beomus')
    wandb.config.update(args)

    checkpoints = Path('training_checkpoints')
    checkpoints.mkdir(exist_ok=True)

    save_path = checkpoints / f'gpt_{config["data"]}_{config["id"]}'
    save_path.mkdir(exist_ok=True)

    batch_size = config['batch_size']
    epochs = config['epochs']
    warmup_steps = 1e2
    sample_every = config['sample_every']
    lr = config['learning_rate']

    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2', 
        bos_token='<|startoftext|>', 
        eos_token='<|endoftext|>', 
        pad_token='<|pad|>'
    )

    max_length = config['max_length']

    dataset = GptData(config['data'], tokenizer, max_length=max_length)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f'There are {train_size} samples for training, and {val_size} samples for validation testing')

    train_dataloader = DataLoader(
        train_dataset,  
        sampler = RandomSampler(train_dataset), # Sampling for training is random
        batch_size = batch_size
    )

    validation_dataloader = DataLoader(
        val_dataset, 
        sampler = SequentialSampler(val_dataset), # Sampling for validation is sequential as the order doesn't matter.
        batch_size = batch_size 
    )

    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    model.resize_token_embeddings(len(tokenizer))
    if config['gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.cuda()
        print(f'Current device: {torch.cuda.current_device()}.')
    else:
        device = 'cpu'

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = warmup_steps, 
        num_training_steps = total_steps
    )

    total_t0 = time.time()
    training_stats = []

    model = model.to(device)

    for e in range(epochs):
        t0 = time.time()
        total_train_loss = 0
        model.train()
        
        tqdm_loader = tqdm(train_dataloader, total=len(train_dataloader))
        for step, batch in enumerate(tqdm_loader):
            tqdm_loader.set_description(f'Epoch {e}')

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            # instead of model.zero_grad
            for p in model.parameters():
                p.grad = None

            outputs = model(
                b_input_ids,
                labels=b_labels, 
                attention_mask = b_masks,
                token_type_ids=None
            )

            loss = outputs[0]

            batch_loss = loss.item()
            wandb.log({'loss': loss})
            total_train_loss += batch_loss

            # Get sample every 100 batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print(f'\nBatch {step} of {len(train_dataloader)}. Loss:{batch_loss}. Time:{elapsed}')

                model.eval()

                sample_outputs = model.generate(
                    bos_token_id=random.randint(1,30000),
                    do_sample=True,   
                    top_k=50, 
                    max_length = 200,
                    top_p=0.95, 
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print(f'Example output: {tokenizer.decode(sample_output, skip_special_tokens=True)}')
                
                model.train()
                        
            loss.backward()
            optimizer.step()
            scheduler.step()


        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)       
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print(f'Average Training Loss: {avg_train_loss}.')

        t0 = time.time()

        model.eval()

        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            
            with torch.no_grad():        

                outputs  = model(
                    b_input_ids,  
                    attention_mask=b_masks,
                    labels=b_labels
                )
            
                loss = outputs[0]  
                
            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        validation_time = format_time(time.time() - t0)    

        print(f'Validation loss: {avg_val_loss}. Validation Time: {validation_time}')

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': e + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        wandb.log(
            {
                'Average Training Loss': avg_train_loss,
                'Average Validation Loss': avg_val_loss
            }
        )

    print(f'Total training took {format_time(time.time()-total_t0)}')

    # See here for serialization: https://huggingface.co/transformers/v1.2.0/serialization.html
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(args, os.path.join(save_path, 'training_args.bin'))


if __name__ == "__main__":
    main()
