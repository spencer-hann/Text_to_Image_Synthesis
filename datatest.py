from dataset import Birds
from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

data = Birds()

test_num = 4300

test = data[test_num]
print(test[1:]) #don't print image tensor
to_pil = transforms.ToPILImage()
img = to_pil(test[0])
#print(data.descriptions[:10])
img.show()
#plt.imshow(test[0].numpy(), cm)
