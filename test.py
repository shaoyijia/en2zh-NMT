import torch
from model import Encoder, Decoder, Seq2Seq
from datahelper import Dataloader
import pdb
from utils import calculate_bleu

# hyper parameters
max_len = 128
batch_size = 64
hid_dim = 256
embed_dim = 256
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
print(device)


def inference(model, iterator, en_field, zh_field):
    model.eval()

    with torch.no_grad():
        predict_res = []
        for _, batch in enumerate(iterator):
            src, src_len = batch.en
            id, trg = batch.id, batch.zh
            src = src.data.to(device)
            trg = trg.data.to(device)
            # 得到decoder的输出结果
            output = model(src, teacher_forcing_ratio=0.0)
            # pdb.set_trace()
            for sent in range(src.shape[1]):
                if en_field is not None:
                    eos_index = [x.item() for x in src[:, sent]].index(en_field.vocab.stoi['<eos>'])
                    src_str = ' '.join([en_field.vocab.itos[x.item()] for x in src[1: eos_index, sent]])
                    sent_id = id[sent]
                predicts = []
                grounds = []

                # 请补充生成部分的代码
                # greedy search
                for di in range(1, max_len):
                    s = zh_field.vocab.itos[torch.argmax(output[di, sent])]
                    if s == '<eos>' or s == '<pad>':
                        break
                    predicts.append(s)
                for di in range(1, trg.shape[0]):
                    s = zh_field.vocab.itos[trg[di][sent]]
                    if s == '<eos>' or s == '<pad>':
                        break
                    grounds.append(s)

                predict_res.append((int(sent_id), src_str, ' '.join(predicts), " ".join(grounds)))

    # (src, prediction, gt)
    predict_res = [(item[1], item[2], item[3]) for item in sorted(predict_res, key=lambda x: x[0])]
    # (prediction, gt)
    bleu = calculate_bleu([i[1] for i in predict_res], [i[2] for i in predict_res])
    return bleu, predict_res


def main():
    test_loader = Dataloader(batch_size, device, eval=True)
    input_dim = len(test_loader.en_field.vocab)
    output_dim = len(test_loader.zh_field.vocab)

    # load the model
    encoder = Encoder(input_dim, embed_dim, hid_dim, n_layers=2, dropout=0)
    decoder = Decoder(embed_dim, hid_dim, output_dim, n_layers=1, dropout=0)
    seq2seq = Seq2Seq(encoder, decoder, device).to(device)
    seq2seq.load_state_dict(torch.load("./model/seq2seq_params.pkl"))

    bleu, predict_res = inference(seq2seq, test_loader.test_iterator, test_loader.en_field, test_loader.zh_field)
    print('bleu: {}'.format(bleu))
    f = open('result.txt', 'w')
    for result in predict_res:
        f.write("{}, {}, {}\n".format(result[0], result[1], result[2]))
    f.close()


if __name__ == "__main__":
    main()
