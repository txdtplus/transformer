import torch
import torch.nn as nn
import torch.nn.functional as F
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class Transformer(nn.Module):
    def __init__(self, num_embeddings=100, N=6, d_model=512, max_length=100, num_heads=8, dropout=0.1, d_ff=2048):
        super(Transformer, self).__init__()

        r"""The parameters of the model.

            num_embeddings (int): The number of the words in the vocabulary.
            d_model (int): The dimension of each word vector.
        """

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.encoder = Encoder(N, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(N, d_model, num_heads, d_ff, dropout)
        self.output_layer = nn.Linear(in_features=d_model, out_features=num_embeddings)

    def forward(self, input_sequence, target_sequence) -> torch.Tensor:
        r"""The inputs of the model.

            input_sequence (Tensor): [batch_size, sentence_length]
            target_sequence (Tensor): [batch_size, sentence_length]
        """

        input_embedding = self.embedding(input_sequence)        # [batch_size, sentence_length] -->[batch_size, sentence_length, d_model]
        target_embedding = self.embedding(target_sequence)      # [batch_size, sentence_length] -->[batch_size, sentence_length, d_model]

        input_embedding = self.positional_encoding(input_embedding)
        target_embedding = self.positional_encoding(target_embedding)

        encoder_output = self.encoder(input_embedding)
        decoder_output = self.decoder(target_embedding, encoder_output)

        output = self.output_layer(decoder_output)      # [batch_size, sentence_length, num_embeddings]

        return output.transpose(-1, -2)                 # [batch_size, num_embeddings, sentence_length]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_length=1000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.position_encoding = self.create_position_encoding(d_model, max_length)
 
    def create_position_encoding(self, d_model, max_length):
        position_encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2, dtype=float) * math.log(10000) / d_model)
        tmp = position * div_term
        position_encoding[:, 0::2] = torch.sin(tmp)
        position_encoding[:, 1::2] = torch.cos(tmp)
        return position_encoding.unsqueeze(0)
    
    def forward(self, x:torch.Tensor):
        return x + self.position_encoding[:, :x.size(1), :].to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=6):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dk = d_model // num_heads
        self.dv = d_model // num_heads

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor):
        # input: Q: [batch_size, sentence_length, d_model]

        batch_size = Q.size(0)

        query = self.query_linear(Q).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)      # [batch_size, num_heads, sentence_length, dk]
        key = self.key_linear(K).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)          # [batch_size, num_heads, sentence_length, dk]
        value = self.value_linear(V).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)

        # value2 = self.value_linear(V).view(batch_size, self.num_heads, -1, self.dk)
        # 由于num_heads和dk是将d_model扩展成两个维度，所以这两个放在后面，然后转置
        # 因此value2这种写法不对

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dk)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)       
        # contiguous 的作用是将张量内存存储变为连续存储。如果没有这个view()会报错，view()在这里相当于concat的作用

        attention_output = self.out_linear(attention_output)

        return attention_output


class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()

        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),      # 根据dropout的原理，这个一层肯定要放到神经元后边
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.FFN(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=6, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)


    def forward(self, x):
        r"""The inputs of the EncoderLayer.

            x (Tensor): [batch_size, sentence_length, d_model]
        """

        residual = x
        x = self.multi_head_attention(x, x, x)
        x = self.dropout(x)
        x = self.layer_norm1(residual + x)

        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.layer_norm2(residual + x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=6, d_ff=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention2 = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, encoder_output):
        r"""The inputs of the EncoderLayer.

            x (dec_embeddings, Tensor): [batch_size, src_sentence_length, d_model]
            encoder_output (Tensor): [batch_size, tgt_sentence_length, d_model]
        """

        residual = x
        x = self.multi_head_attention1(x, x, x)
        x = self.dropout(x)
        x = self.layer_norm1(residual + x)

        residual = x
        x = self.multi_head_attention2(x, encoder_output, encoder_output)
        x = self.dropout(x)
        x = self.layer_norm2(residual + x)

        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.layer_norm3(residual + x)

        return x

  
class Encoder(nn.Module):
    def __init__(self, N = 6, d_model=512, num_heads=6, d_ff=2048, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout)
              for _ in range(N)]
            )
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, N = 6, d_model=512, num_heads=6, d_ff=2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout)
              for _ in range(N)]
            )
    
    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x


if __name__ == "__main__":
    input = torch.tensor([[1,2,3], [4,5,6]]).to(device)  # input_sequence: [batch_size, sentence_length]
    target = torch.tensor([[2,3], [5,6]]).to(device)     # output_sequence: [batch_size, sentence_length]
    model = Transformer(num_embeddings=10)
    model.to(device)

    output = model(input, target)           # model_output: [batch_size, num_embeddings, sentence_length]
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)



    