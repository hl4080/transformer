from conf import *
from data import *
import torch


model = torch.load('model.pth')

def evaluate(inp_sentence, model):
    decoder_input = torch.tensor([src_vocab["<bos>"]]).unsqueeze(0).to(device)
    for i in range(max_len):
        predictions, _, _, _ = model(torch.LongTensor(inp_sentence).view(1, -1).to(device), decoder_input)
        predictions = predictions[-1:, :] #最后一个时间步的预测结果
        predicted_id = torch.argmax(predictions, dim=-1).to(device) #获取概率最高的索引
        if predicted_id == tgt_vocab["<eos>"]:
            break
        predicted_id = predicted_id.unsqueeze(0).to(device)
        decoder_input = torch.cat([decoder_input, predicted_id], dim=-1).to(device) #将生成的词汇索引倒入输入中
    
    return decoder_input.squeeze().tolist() #生成的序列去掉多余维度

while True:
    sentence = input("Enter a sentence: ")
    if sentence.lower() == "exit":
        break
    inp_sentence = [tgt_vocab["<sos>"]] + [src_vocab[token] for token in tokenizer.tokenize_de(sentence)] + [tgt_vocab["<eos>"]]
    result = evaluate(inp_sentence, model)
    result = [tgt_vocab.get_itos()[idx] for idx in result]
    result = " ".join(result)
    print(f"Response: {result}")
