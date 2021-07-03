import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from test import inference
from datahelper import Dataloader
from tensorboardX import SummaryWriter


# hyper parameters
n_epochs = 100
batch_size = 64
hid_dim = 256
embed_dim = 256
dropout = 0.2
lr = 0.0001
clip = 10  # in case of gradient explosion
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
print(device)

writer = SummaryWriter('./train_log2')
f = open('loss2.txt', 'w')


def evaluate(model, val_iter, vocab_size, pad):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for b, batch in enumerate(val_iter):
            src, len_src = batch.en
            trg = batch.zh
            src = src.data.to(device)
            trg = trg.data.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            # negative log likelihood loss
            loss = F.nll_loss(output[1:].view(-1, vocab_size),
                              trg[1:].contiguous().view(-1),
                              ignore_index=pad)
            total_loss += loss.data.item()

        return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, pad):
    model.train()
    total_loss = 0
    for b, batch in enumerate(train_iter):
        src, len_src = batch.en
        trg = batch.zh
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        # negative log likelihood loss
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                          trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        loss.backward()
        # avoid gradient explosion
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data.item()

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("batch: {}, loss: {}".format(b, total_loss))
            f.write("epoch: {}, train_loss: {}\n".format(e, total_loss))
            writer.add_scalar('train_loss', total_loss, e)
            total_loss = 0


def main():
    train_loader = Dataloader(batch_size, device, eval=False)
    input_dim = len(train_loader.en_field.vocab)
    output_dim = len(train_loader.zh_field.vocab)

    # declare the network
    encoder = Encoder(input_dim, embed_dim, hid_dim, n_layers=2, dropout=dropout)
    decoder = Decoder(embed_dim, hid_dim, output_dim, n_layers=1, dropout=dropout)
    seq2seq = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(seq2seq.parameters(), lr=lr)

    best_val_loss = 10000000
    best_bleu = 0
    for epoch in range(n_epochs):
        train(epoch, seq2seq, optimizer, train_loader.train_iterator,
              output_dim, clip, train_loader.zh_field.vocab.stoi['<pad>'])
        # test on the validation set
        val_loss = evaluate(seq2seq, train_loader.dev_iterator, output_dim, train_loader.zh_field.vocab.stoi['<pad>'])
        bleu, _ = inference(seq2seq, train_loader.dev_iterator, train_loader.en_field, train_loader.zh_field)

        # save the best model
        if val_loss < best_val_loss:
            torch.save(seq2seq.state_dict(), './model/seq2seq_best_loss.pkl')
            best_val_loss = val_loss
        if bleu > best_bleu:
            torch.save(seq2seq.state_dict(), './model/seq2seq_best_bleu.pkl')
            best_bleu = bleu
        print("Best val_loss: {}, Best bleu: {}, Epoch: {}, val_loss: {}, bleu: {}".format(
            best_val_loss, best_bleu, epoch, val_loss, bleu))
        f.write("Best val_loss: {}, Best bleu: {}, Epoch: {}, val_loss: {}, bleu: {}\n".format(
            best_val_loss, best_bleu, epoch, val_loss, bleu))

        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('bleu', bleu, epoch)

    writer.close()
    f.close()


if __name__ == "__main__":
    main()
