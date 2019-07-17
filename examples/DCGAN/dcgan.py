from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Root directory for dataset
dataroot = "/home/achang/Workspace/deeplearning/datasets/celeba"
# Number of workers for dataloader
workers = 4
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                           transforms.Resize(image_size),
                           transforms.CenterCrop(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True,num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu" )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nz,ngf*6,4,1,0,bias=False),
                nn.BatchNorm2d(ngf*6),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf*6,ngf*4,4,2,1,bias=False),
                nn.BatchNorm2d(ngf*4),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
                nn.BatchNorm2d(ngf*2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf,nc,4,2,1,bias=False),
                nn.Tanh()
                )
    def forward(self,input):
        return self.main(input)

netG = Generator(ngpu).to(device)

if (device.type=='cuda') and (ngpu>1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4,2,1,bias=False),
                nn.ReLU(),

                nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
                nn.BatchNorm2d(ndf*2),
                nn.ReLU(),

                nn.Conv2d(ndf*2, ndf*4, 4,2,1,bias=False),
                nn.BatchNorm2d(ndf*4),
                nn.ReLU(),

                nn.Conv2d(ndf*4, ndf*8, 4,2,1,bias=False),
                nn.BatchNorm2d(ndf*8),
                nn.ReLU(),

                nn.Conv2d(ndf*8, 1, 4,1,0,bias=False),
                nn.Sigmoid()
                )

    def forward(self,input):
        return self.main(input)


netD = Discriminator(ngpu).to(device)
if(device.type=='cuda') and (ngpu>1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

#binary cross entropy loss (BCELoss)
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1,1,device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0
print("Starting Training Loop...")
for epoch in range(num_epochs):

    for i, data in enumerate(dataloader, 0):
        #discriminator training
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,),real_label, device=device)#create a vector of 1s
        output=netD(real_cpu).view(-1)#forward pass
        errD_real = criterion(output,label)
        errD_real.backward()#backward pass
        D_x = output.mean().item()#average output values only for print

        noise=torch.randn(b_size,nz,1,1,device=device)
        fake=netG(noise)#forward random values to generator to create fake images
        label.fill_(fake_label)
        output=netD(fake.detach()).view(-1)#forward to discriminator
        errD_fake=criterion(output,label)
        errD_fake.backward()#backward
        D_G_z1 = output.mean().item()
        errD = errD_real+errD_fake
        optimizerD.step()

        #generator training
        netG.zero_grad()
        label.fill_(real_label)
        output=netD(fake).view(-1)
        errG=criterion(output,label)
        errG.backward()
        D_G_z2=output.mean().item()
        optimizerG.step()

        if i%50==0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if(iters%500==0) or ((epoch==num_epochs-1) and (i==len(dataloader)-1)):
            with torch.no_grad():
                fake=netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake,padding=2,normalize=True))
        iters+=1

#save checkpoint
torch.save({
'discriminator':netD.state_dict,
'generator':netG.state_dict,
'optimizerD':optimizerD,
'optimizerG':optimizerG
},'dcgan_epoch.pth')

#img = torch.FloatTensor(1, 100, 1, 1).cuda()
img = torch.FloatTensor(1, 100, 1, 1)
torch.onnx.export(netG, img, 'generator.onnx')

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
