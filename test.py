import pickle
import torch

data = torch.load('metadata.pth')

with open('cubs captions.txt', 'r') as file:
    for line in file:
        for word in line.split():
            if word == '0':
                continue
            print(data['word_id_to_word'][int(word)], end=" ")
        print()

