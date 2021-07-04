import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules.transformer import Transformer
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from Model import Discriminator,Generator,initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
IMG_CHANNELS= 3
NOISE_DIM = 100
NUM_EPOCHS = 10
DISC_FEATURES= 64
GEN_FEATURES = 64
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)


transforms = transforms.Compose(
    [transforms.Resize(IMAGE_SIZE),transforms.ToTensor(),transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)])])

#dataset = datasets.fashion_MNIST(root = 'D:\college_notes\ML _DL\Projects\GANs\Celeb_DCGan\datasets',train = True, transform = transforms,download = True,)
dataset = datasets.ImageFolder(root="D:\college_notes\ML _DL\Projects\GANs\Celeb_DCGan", transform=transforms)

dataloader = DataLoader(dataset,batch_size = BATCH_SIZE,shuffle =True)
gen = Generator(NOISE_DIM,IMG_CHANNELS,GEN_FEATURES,).to(device)
disc = Discriminator(IMG_CHANNELS,DISC_FEATURES).to(device)
initialize_weights(gen)
initialize_weights(disc)

gen_opt = optim.Adam(gen.parameters(),lr= LEARNING_RATE,betas = (0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

#For TensorBoard
writer_real = SummaryWriter(f"d:/college_notes/ML_DL/Projects/GANs/Celeb_DCGan/logs/real")
writer_fake = SummaryWriter(f"d:/college_notes/ML_DL/Projects/GANs/Celeb_DCGan/logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_id ,(real,data_fake) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE,NOISE_DIM,1,1).to(device)
        fake = gen(noise)

        ## Discriminator Training max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        gen_opt.step()

        ##Tensor-Board code 
        if batch_id % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_id}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1  
        