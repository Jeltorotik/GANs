import numpy as np
import matplotlib.pyplot as plt

from model import init_gan_architecture
import mygan

from torchvision import transforms, datasets



image_size = 256
batch_size = 64
params_path = ''

transform=transforms.Compose([
	transforms.Resize(image_size),
	transforms.CenterCrop(image_size),
	transforms.ToTensor(),

	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	# Normalize здесь приводит значения в промежуток [-1, 1]
])

dataset = datasets.CelebA('data', download=True, transform=transform)



# Gan initialization:

gan = mygan.GAN(*init_gan_architecture())

gan.download_params(params_path)

gan.sample_and_plot(16)

#gan.add_dataset(dataset, batch_size=batch_size)

#gan.train(saving=True)




