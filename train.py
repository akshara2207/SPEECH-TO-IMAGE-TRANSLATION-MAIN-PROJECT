import json
import pickle

from matplotlib import pyplot as plt, image as mpimg
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

flower_caps = json.load(open('train.json'))
data = flower_caps['data']
bird_caps = json.load(open('train(1).json'))
data += bird_caps['data']
# Flowers is a list of tuples containing the image name and the text converted into an embedding
flowers_and_birds = []
total = len(data)
i = 0
for item in data:
    flowers_and_birds.append((item['img'], model.encode(item['text'], convert_to_tensor=True)))
    print(f"\r{i}/{total} {round((i/total) * 100, 2)}%", end="")
    i += 1
pickle.dump(flowers_and_birds, open("model.pkl", "wb")) # Save the model
print("\nModel saved successfully.")


# For testing
if __name__ == "__main__":
    text = "black bird" # Enter some text here
    embedding = model.encode(text, convert_to_tensor=True)

    flowers_and_birds.sort(key=lambda x: util.cos_sim(embedding, x[1]).sum(), reverse=True)

    _, ax = plt.subplots(2, 2)
    for i in flowers_and_birds[:4]:
        idx = flowers_and_birds.index(i)
        ax[idx//2, idx%2].set_title(i[0])
        ax[idx//2, idx%2].axis("off")
        ax[idx//2, idx%2].imshow(mpimg.imread(i[0]))
    plt.show()
