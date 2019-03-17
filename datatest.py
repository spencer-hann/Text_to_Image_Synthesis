from dataset import Birds
from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

<<<<<<< HEAD
data = Birds(descriptions_per_image=2)
=======
def simple_test():
    data = Birds(descriptions_per_image=2)
>>>>>>> spencer

    test_num = 4300

<<<<<<< HEAD
test = data[test_num]
test = data.get_full_item(test_num)
print(test[1:]) #don't print image tensor
to_pil = transforms.ToPILImage()
img = to_pil(test[0])
#print(data.descriptions[:10])
img.show()
#plt.imshow(test[0].numpy(), cm)
=======
    test = data[test_num]
    test = data.get_full_item(test_num)
    print(test[1:]) #don't print image tensor
    to_pil = transforms.ToPILImage()
    img = to_pil(test[0])
    #print(data.descriptions[:10])
    img.show()
    #plt.imshow(test[0].numpy(), cm)

if __name__ == "__main__":
    simple_test()
>>>>>>> spencer
