import torch
import torch.utils.data as Data
from datasets import TestData, MyTranslationDataSet
from transformer import Transformer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model:Transformer, test_data:TestData):
    # Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    
    dataset = MyTranslationDataSet(raw_data=test_data)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=1)
    output = []
    for data in dataloader:
        enc_input = data[0].to(device)
        dec_input = data[1].to(device)
        output_length = 1
        dec_next = torch.tensor([[1]])

        while output_length < test_data.max_length and torch.ne(dec_next[0,-1], 2):
            dec_output = model(enc_input, dec_input)   # [batch_size, num_embeddings, sentence_length]
            dec_output = dec_output.permute(0, 2, 1)
            dec_next = torch.argmax(dec_output[0,-1], dim=-1).reshape(1, -1)
            dec_input = torch.cat([dec_input, dec_next], dim=1)
            output_length += 1
        output.append(dataset.idx2word(dec_input))
    return output


if __name__ == "__main__":
    model_path = os.getenv("USERPROFILE") + '\.cache\my_transformer\model.pth'
    test_data = TestData()

    model = Transformer(
        num_embeddings=len(test_data.tgt_dict),
        max_length=100
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device=device)
    model.eval()

    output = test(model=model, test_data=test_data)

    for i in range(len(test_data.sentences)):
        print("输入文本为："+test_data.sentences[i][0])
        print("输出文本为："+output[i]+'\n')