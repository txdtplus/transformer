# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch
import torch.utils.data as Data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RawDataset:
    def __init__(self, max_length=100) -> None:
        
        self.src_dict = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8, '爱': 9, '中': 10, '国': 11}  # 词源字典  字：索引
        self.tgt_dict = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9, 'love': 10, 'China': 11}
        # Encoder_input    Decoder_input        Decoder_output
        self.max_length = max_length
        self.reverse_dict()
    
    def reverse_dict(self):
        self.src_reverse_dict = {value:key for key,value in self.src_dict.items()}
        self.tgt_reverse_dict = {value:key for key,value in self.tgt_dict.items()}


class TrainData(RawDataset):
    def __init__(self, max_length=100) -> None:
        super(TrainData, self).__init__(max_length=max_length)
        self.sentences = [
             ['我 是 学 生 P', 'S I am a student', 'I am a student E'],         # S: 开始符号 1,2,3,4,0
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号 1,5,6,3,7
             ['我 是 男 生 P', 'S I am a boy', 'I am a boy E'],                 # 1,2,8,4,0
             ['我 爱 中 国 P', 'S I love China P', 'I love China P E']          # 1,9,10,11,0
             ]                 # P: 占位符号，如果当前句子不足固定长度用P占位


class TestData(RawDataset):
    def __init__(self, max_length=100) -> None:
        super(TestData, self).__init__(max_length=max_length)
        self.sentences = [
            #  ['我 是 学 生 P', 'S I am a student', 'I am a student E'],         # S: 开始符号 1,2,3,4,0
            #  ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号 1,5,6,3,7
            #  ['我 是 男 生 P', 'S I am a boy', 'I am a boy E'],                 # 1,2,8,4,0
             ['我 是 男 生 P', 'S ', ' '],          # 1,9,10,11,0
             ['我 是 学 生 P', 'S ', ' ']
             ]                 # P: 占位符号，如果当前句子不足固定长度用P占位


# 自定义数据集函数
class MyTranslationDataSet(Data.Dataset):
    def __init__(self, raw_data:TestData):
        super(MyTranslationDataSet, self).__init__()
        self.sentences = raw_data.sentences
        self.src_dict = raw_data.src_dict
        self.tgt_dict = raw_data.tgt_dict
        self.tgt_reverse_dict = raw_data.tgt_reverse_dict

        self.src_idx2word = {self.src_dict[key]: key for key in self.src_dict}
        self.tgt_idx2word = {self.tgt_dict[key]: key for key in self.tgt_dict}

        self.src_vocab_size = len(self.src_dict)
        self.tgt_vocab_size = len(self.tgt_dict)

        self.src_len = len(self.sentences[0][0].split(" "))               # Encoder输入的最大长度
        self.tgt_len = len(self.sentences[0][1].split(" "))               # Decoder输入输出最大长度

        self.enc_inputs, self.dec_inputs, self.dec_outputs = self.word2idx()

    def word2idx(self):
        enc_inputs, dec_inputs, dec_outputs = [], [], []
        for i in range(len(self.sentences)):
            enc_input = [self.src_dict[n] for n in self.sentences[i][0].split()]
            dec_input = [self.tgt_dict[n] for n in self.sentences[i][1].split()]
            dec_output = [self.tgt_dict[n] for n in self.sentences[i][2].split()]
            enc_inputs.append(enc_input)
            dec_inputs.append(dec_input)
            dec_outputs.append(dec_output)
        return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
    
    def idx2word(self, dec_input:torch.Tensor) -> str:
        dec_input = dec_input.view(-1)
        output_list = dec_input.view(-1).cpu().numpy().tolist()

        if self.tgt_dict['S'] in output_list:
            output_list.remove(self.tgt_dict['S'])
        if self.tgt_dict['P'] in output_list:
            output_list.remove(self.tgt_dict['P'])
        if self.tgt_dict['E'] in output_list:
            output_list.remove(self.tgt_dict['E'])

        output_word_list = [self.tgt_reverse_dict[idx] for idx in output_list]
        output = " ".join(output_word_list)
        return output

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
    


if __name__ == "__main__":

    raw_data = TrainData()
    dataset = MyTranslationDataSet(raw_data=raw_data)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)

    for epoch in range(5):
        for batch_idx, data in enumerate(dataloader):
            
            enc_inputs = data[0].to(device)
            dec_inputs = data[1].to(device)
            dec_outputs = data[2].to(device)
            # print(batch_idx)

