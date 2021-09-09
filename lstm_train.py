import argparse
import json
from pathlib import Path
import time
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from utils import LstmData, format_time
import model


def parse():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('-d', '--data', type=str, choices=['lotr', 'hp'],
                        default='lotr', help='dataset for training')
    parser.add_argument('-n', '--n_pred', type=int, default=9,
                        help='number of words used in prediction')
    parser.add_argument('-o', '--occurence', type=int, default=10,
                        help='number of occurences of a word to be added in vocab.')
    
    # hyperparams
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=16, help='epochs')
    parser.add_argument('-c', '--clip', type=int, default=1, 
                        help='clipping to avoid exploding gradient.')

    # model params
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--layer_norm', action='store_true', default=False)
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='entire model checkpoint to resume training.')

    # misc.
    parser.add_argument('--save_interval', type=int, default=4)
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

    save_path = checkpoints / f'{config["data"]}_{config["id"]}'
    save_path.mkdir(exist_ok=True)

    n_pred = config['n_pred']
    min_occurences = config['occurence']
    batch_size = config['batch_size']

    lstm_data = LstmData(
        dataset=config['data'], 
        min_occurences=min_occurences, 
        n_pred=n_pred
    )
    word_to_id = lstm_data.word_to_id
    training_dataset = lstm_data.train_data

    training_loader = DataLoader(
        training_dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )


    vocab_size = len(word_to_id) + 1
    embedding_dim = config['embed_dim']  # size of the word embeddings
    hidden_dim = config['hidden_dim']    # size of the hidden state
    n_layers = config['layers']          # number of LSTM layers
    layer_norm = config['layer_norm']

    epochs = config['epochs']
    learning_rate = config['learning_rate']
    clip = config['clip']

    if config['resume']:
        net = torch.load(config['resume'], map_location='cpu')
    else:
        net = model.LSTM(vocab_size, embedding_dim, hidden_dim, n_layers, layer_norm=layer_norm)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    if config['gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Current device: {torch.cuda.current_device()}.')
    else:
        device = 'cpu'

    net = net.to(device=device)
    
    # starting training
    net.train()
    start_time = time.time()
    print("Starting training")
    for e in range(epochs):
        hidden = net.init_hidden(batch_size)
        hidden = (hidden.to(device) for hidden in hidden)

        # loops through each batch
        tqdm_loader = tqdm(training_loader, total=len(training_loader))
        for features, labels in tqdm_loader:
            tqdm_loader.set_description(f'Epoch {e}')

            # resets training history
            hidden = tuple([each.data for each in hidden])
            # net.zero_grad()
            for p in net.parameters():
                p.grad = None

            # computes gradient of loss from backprop
            features = features.to(device)
            labels = labels.to(device)
            output, hidden = net.forward(features, hidden)
            loss = loss_func(output, labels)
            loss.backward()

            # using clipping to avoid exploding gradient
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            # scheduler.step()

            wandb.log({'loss': loss})
        
        if e % config['save_interval'] == 0:
            torch.save(net, f'{save_path}/checkpoint_{e}.pth')


    net.eval()
    torch.save(net, f'{save_path}/last_checkpoint.pth')

    save_misc = save_path / 'misc'
    save_misc.mkdir(exist_ok=True)
    with open(f'{save_misc}/word_to_id.json', 'w') as fp:
        json.dump(word_to_id, fp, indent=4)

    start_time = time.time()
    print(f"Training took: {format_time(time.time() - start_time)}")


if __name__ == "__main__":
    main()
