import os
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningModule):
    def __init__(self, data_dir="./data", batch_size=256, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_worksers = num_workers

        self.trasform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transforms)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=self.num_worksers
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_worksers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_worksers
        )


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        print(X.shape)
        X = self.relu(self.max_pool(self.conv1(X)))
        X = self.relu(self.max_pool(self.conv_drop(self.conv2(X))))
        X = X.view(-1, 320)
        X = self.relu(self.fc1(X))
        X = self.fc2(X)
        return self.sigmoid(X)


class Generator(nn.Module):
    def __init__(self, latent_dim) -> None:
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(latent_dim, 7 * 7 * 64)
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.conv = nn.ConvTranspose2d(16, 1, 4, stride=2)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.relu(self.lin1(X))
        X = X.view(-1, 64, 7, 7)
        X = self.relu(self.ct1(X))
        X = self.relu(self.ct2(X))
        X = self.conv(X)
        return X


class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

        self.validation_z = torch.randn(6, latent_dim)

    def forward(self, X):
        X = self.generator(X)
        print(X.shape)
        return X

    def adversarial_loss(self, Y_hat, Y):
        return F.binary_cross_entropy(Y_hat, Y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        print(real_imgs.shape)

        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)

        if optimizer_idx == 0:
            fake_imgs = self(z)
            y_hat = self.discriminator(fake_imgs)

            y = torch.ones(real_imgs.size(0), 1)

            g_loss = self.adversarial_loss(y_hat, y)

            log_dict = {"g_loss": g_loss}
            return {"loss": g_loss, "progress_bar": log_dict}
        if optimizer_idx == 1:
            y_hat_real = self.discriminator(real_imgs)
            y_hat = torch.ones(real_imgs.size(0), 1)
            real_loss = self.adversarial_loss(y_hat_real, y_hat)

            y_hat_fake = self.discriminator(self(z).detach())
            y_fake = torch.zeros(real_imgs.size(0), 1)

            fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

            d_loss = (real_loss + fake_loss) / 2
            log_dict = {"d_loss": d_loss}
            return {"loss": d_loss, "progress_bar": log_dict}

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.lin1.weight)
        sample_imgs = self(z)
        print(f"epoch: {self.current_epoch}")
        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(
                sample_imgs.detach()[i, 0, :, :], cmap="gray_r", interpolation=None
            )
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
        plt.show()

    def on_epoch_end(self):
        self.plot_imgs()


dm = MNISTDataModule()
model = GAN()
model.plot_imgs()
trainer = pl.Trainer(max_epochs=20)
print("starting")
trainer.fit(model, dm)
