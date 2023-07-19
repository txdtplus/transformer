# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import TrainData, MyTranslationDataSet
from transformer import Transformer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    raw_data = TrainData()
    dataset = MyTranslationDataSet(raw_data=raw_data)
    dataloader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)

    model = Transformer(
        num_embeddings=len(raw_data.tgt_dict),
        max_length=100
    )
    model.to(device=device)
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=0)         # 忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(60):
        for batch_idx, data in enumerate(dataloader):   # enc_inputs : [batch_size, sentence_length]
                                                        # dec_inputs : [batch_size, sentence_length]
                                                        # dec_outputs: [batch_size, sentence_length]
            enc_inputs = data[0].to(device)
            dec_inputs = data[1].to(device)
            dec_outputs = data[2].to(device)
            
            outputs = model(enc_inputs, dec_inputs)     # outputs: [batch_size, num_embeddings, sentence_length]
            loss = criterion(outputs, dec_outputs)
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 保存模型到用户缓存路径       
    write_folder = os.getenv("USERPROFILE") + '\.cache\my_transformer'
    if not os.path.exists(write_folder):
        os.mkdir(write_folder)

    torch.save(model.state_dict(), os.path.join(write_folder, 'model.pth'))
    print("保存模型")
