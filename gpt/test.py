from conf import *
from data import *
import torch


model = torch.load('model.pth')

def evaluate(inp_sentence, model):
    src = inp_sentence
    for i in range(max_len-1):
        predictions, _ = model(src)
        predictions = predictions[-1:, :] #最后一个时间步的预测结果
        predicted_id = torch.argmax(predictions, dim=-1).to(device) #获取概率最高的索引
        if predicted_id == vocab['<eos>']:
            break
        src = torch.cat([src, predicted_id.unsqueeze(0)], dim=0).to(device) #将生成的词汇索引倒入输入中
    return src.squeeze().tolist() #生成的序列去掉多余维度

while True:
    sentence = input("Enter a sentence: ")
    if sentence.lower() == "exit":
        break
    inp_sentence = torch.tensor([vocab['<bos>']] + vocab(tokenizer(sentence)), dtype=torch.long).unsqueeze(1).to(device)
    result = evaluate(inp_sentence, model)
    result = [vocab.get_itos()[idx] for idx in result][inp_sentence.size(0):]
    result = " ".join(result)
    print(f"Response: {result}")
