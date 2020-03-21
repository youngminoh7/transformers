import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torchtext import data, datasets 
from module import ClassificationTransformer 

from argparse import ArgumentParser 
from tqdm import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def train(data_iterator, model, optimizer, criterion, args): 
    n_sample, n_correct = 0, 0
    batch_loss = 0.0

    for batch in tqdm(data_iterator):
        optimizer.zero_grad()

        input = batch.text[0]
        label = batch.label - 1

        if input.size(1) > args.max_len:
            input = input[:, :args.max_len]
        out = model(input)

        n_sample += input.size(0)
        n_correct += (label == out.argmax(dim=1)).sum().item()

        loss = criterion(out, label)
        batch_loss += loss.item() * input.size(0)
        loss.backward()

        optimizer.step()

    print('train loss: {:.3f}, accuracy: {:.3f}'.format(batch_loss / n_sample, 
                                                        n_correct / n_sample))


def validate(data_iterator, model, criterion, args):
    with torch.no_grad():
        n_sample, n_correct = 0, 0
        batch_loss = 0.0

        for batch in tqdm(data_iterator):

            input = batch.text[0]
            label = batch.label - 1

            if input.size(1) > args.max_len:
                input = input[:, :args.max_len]
            out = model(input)

            n_sample += input.size(0)
            n_correct += (label == out.argmax(dim=1)).sum().item()

            loss = criterion(out, label)
            batch_loss += loss.item() * input.size(0)
            
    print('test loss: {:.3f}, accuracy: {:.3f}'.format(batch_loss / n_sample, 
                                                       n_correct / n_sample))


def main(args):

    TEXT = data.Field(batch_first=True, lower=True, include_lengths=True)
    LABEL = data.Field(sequential=False, batch_first=True) 

    print("Loading IMDB dataset...")
    trainset, testset = datasets.IMDB.splits(TEXT, LABEL) 

    print("Building vocabulary: {} words...".format(args.vocab_size))
    TEXT.build_vocab(trainset, max_size=args.vocab_size - 2)
    LABEL.build_vocab(trainset) 

    train_iter, test_iter = data.BucketIterator.splits((trainset, testset), 
                                                       batch_size=args.batch, 
                                                       device=device)

    print("Instantiating model...")
    model = ClassificationTransformer(n_layer=args.n_layer, 
                                      n_head=args.n_head, 
                                      d_embed=args.d_embed, 
                                      n_token=args.vocab_size, 
                                      n_sequence=args.max_len, 
                                      n_class=2, 
                                      embedding_pos=args.position_embedding)
    model.to(device) 
    print(model)

    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters()) 
    criterion = nn.NLLLoss() 

    for epoch in range(args.epochs):
        print('epoch', epoch + 1)
        
        model.train(True)
        train(train_iter, model, optimizer, criterion, args)

        model.train(False)
        validate(test_iter, model, criterion, args)

    torch.save(model.state_dict(), 'saved_model.pt')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int) 
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--vocab_size", default=5000, type=int) 
    parser.add_argument("--n_layer", default=1, type=int)
    parser.add_argument("--n_head", default=3, type=int) 
    parser.add_argument("--d_embed", default=16, type=int)
    parser.add_argument("--position_embedding", default=False, type=bool)
    args = parser.parse_args() 

    main(args)
