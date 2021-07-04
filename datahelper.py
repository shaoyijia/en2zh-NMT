# in new version, it should be torchtext.legacy.data
from torchtext.data import Example, Dataset, Field, Iterator, BucketIterator
import dill
from bpe import Encoder

MAX_LEN = 128


class vanilla_Dataloader:
    """
    jupyter notebook中提供的原始版本
    """
    def __init__(self, batch_size, device, eval=False):
        raw_data = self.read_data("./data/", test=eval)
        # 训练模式
        if not eval:
            train_data, dev_data = raw_data
            # 定义数据字段
            self.id_field = Field(sequential=False, use_vocab=False)
            self.en_field = Field(init_token='<sos>', eos_token='<eos>', lower=True, include_lengths=True)
            self.zh_field = Field(init_token='<sos>', eos_token='<eos>', lower=True)
            self.fields = [("id", self.id_field), ("en", self.en_field), ("zh", self.zh_field)]

            # 构建数据集
            train_dataset = Dataset(
                [Example.fromlist([idx, item[0], item[1]], self.fields) for idx, item in enumerate(train_data)],
                self.fields)
            dev_dataset = Dataset(
                [Example.fromlist([idx, item[0], item[1]], self.fields) for idx, item in enumerate(dev_data)],
                self.fields)

            # 构建数据迭代器
            self.train_iterator = BucketIterator(train_dataset, batch_size=batch_size, device=device,
                                                 sort_key=lambda x: len(x.en), sort_within_batch=True)
            self.dev_iterator = BucketIterator(dev_dataset, batch_size=batch_size, device=device,
                                               sort_key=lambda x: len(x.en), sort_within_batch=True)

            # 构建词典
            self.en_field.build_vocab(train_dataset, min_freq=2)
            self.zh_field.build_vocab(train_dataset, min_freq=2)

            # 存储字段
            dill.dump(self.en_field, open("./model/EN.Field", "wb"))
            dill.dump(self.zh_field, open("./model/ZH.Field", "wb"))

            print("en vocab size:", len(self.en_field.vocab.itos), "zh vocab size:", len(self.zh_field.vocab.itos))

        # 测试模式
        else:
            test_data = raw_data[-1]
            # 加载存储的字段
            self.id_field = Field(sequential=False, use_vocab=False)
            self.en_field = dill.load(open("./model/EN.Field", "rb"))
            self.zh_field = dill.load(open("./model/ZH.Field", "rb"))
            self.fields = [("id", self.id_field), ("en", self.en_field), ("zh", self.zh_field)]

            # 构建测试集 & 迭代器
            test_data = Dataset(
                [Example.fromlist([idx, item[0], item[1]], self.fields) for idx, item in enumerate(test_data)],
                self.fields)
            self.test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device, train=False,
                                                sort_key=lambda x: len(x.en), sort_within_batch=True)

    # 从文件中读取数据
    def read_data(self, path, test=True, lang1='en', lang2='zh'):
        data = []
        types = ['test'] if test else ['train', 'dev']
        # print(types)
        for type in types:
            sub_data = []
            with open(f"{path}/{type}.seg.{lang1}.txt", encoding='utf-8') as f1, open(f"{path}/{type}.seg.{lang2}.txt",
                                                                                      encoding='utf-8') as f2:
                for src, trg in zip(f1, f2):
                    if len(src) > MAX_LEN and len(trg) > MAX_LEN:
                        continue
                    sub_data.append((src.strip(), trg.strip()))
            data.append(sub_data)

        return data


class Dataloader:
    """
    使用BPE算法对词表进行优化
    """
    def __init__(self, batch_size, device, eval=False):
        raw_data = self.read_data("./data/", test=eval)
        # 训练模式
        if not eval:
            train_data, dev_data = raw_data
            # 定义数据字段
            self.id_field = Field(sequential=False, use_vocab=False)
            self.en_field = Field(init_token='<sos>', eos_token='<eos>', lower=True, include_lengths=True)
            self.zh_field = Field(init_token='<sos>', eos_token='<eos>', lower=True)
            self.fields = [("id", self.id_field), ("en", self.en_field), ("zh", self.zh_field)]

            # 构建数据集
            train_dataset = Dataset(
                [Example.fromlist([idx, item[0], item[1]], self.fields) for idx, item in enumerate(train_data)],
                self.fields)
            dev_dataset = Dataset(
                [Example.fromlist([idx, item[0], item[1]], self.fields) for idx, item in enumerate(dev_data)],
                self.fields)

            # 构建数据迭代器
            self.train_iterator = BucketIterator(train_dataset, batch_size=batch_size, device=device,
                                                 sort_key=lambda x: len(x.en), sort_within_batch=True)
            self.dev_iterator = BucketIterator(dev_dataset, batch_size=batch_size, device=device,
                                               sort_key=lambda x: len(x.en), sort_within_batch=True)

            # 构建词典
            self.en_field.build_vocab(train_dataset, min_freq=2)
            self.zh_field.build_vocab(train_dataset, min_freq=2)

            # 存储字段
            dill.dump(self.en_field, open("./model/EN.Field", "wb"))
            dill.dump(self.zh_field, open("./model/ZH.Field", "wb"))

            print("en vocab size:", len(self.en_field.vocab.itos), "zh vocab size:", len(self.zh_field.vocab.itos))

        # 测试模式
        else:
            test_data = raw_data[-1]
            # 加载存储的字段
            self.id_field = Field(sequential=False, use_vocab=False)
            self.en_field = dill.load(open("./model/EN.Field", "rb"))
            self.zh_field = dill.load(open("./model/ZH.Field", "rb"))
            self.fields = [("id", self.id_field), ("en", self.en_field), ("zh", self.zh_field)]

            # 构建测试集 & 迭代器
            test_data = Dataset(
                [Example.fromlist([idx, item[0], item[1]], self.fields) for idx, item in enumerate(test_data)],
                self.fields)
            self.test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device, train=False,
                                                sort_key=lambda x: len(x.en), sort_within_batch=True)

    # 从文件中读取数据（加入了对词表的优化）
    # Use BPE algorithm to get subwords
    # reference open source code: https://github.com/soaxelbrooke/python-bpe
    def read_data(self, path, test=True, lang1='en', lang2='zh'):
        data = []
        types = ['test'] if test else ['train', 'dev']
        # print(types)
        words = []
        with open(f"{path}/train.seg.{lang1}.txt", encoding='utf-8') as f1, open(f"{path}/train.seg.{lang2}.txt",
                                                                                  encoding='utf-8') as f2:
            for src, trg in zip(f1, f2):
                if len(src) > MAX_LEN and len(trg) > MAX_LEN:
                    continue
                words.append(src.strip())
        encoder = Encoder()
        encoder.fit(words)

        for type in types:
            sub_data = []
            with open(f"{path}/{type}.seg.{lang1}.txt", encoding='utf-8') as f1, open(f"{path}/{type}.seg.{lang2}.txt",
                                                                                      encoding='utf-8') as f2:
                for src, trg in zip(f1, f2):
                    if len(src) > MAX_LEN and len(trg) > MAX_LEN:
                        continue
                    sub_data.append((list(encoder.tokenize(src.strip())), trg.strip()))
            data.append(sub_data)

        return data
