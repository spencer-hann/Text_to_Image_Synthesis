from dataset import Birds
from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
data = Birds()

test = data[50]
print(test[1])
to_pil = transforms.ToPILImage()
img = to_pil(test[0])
print(data.descriptions[:10])
img.show()
#plt.imshow(test[0].numpy(), cm)
