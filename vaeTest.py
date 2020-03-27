import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from tqdm import tqdm


# create VAE From pytorch example
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(27648, 4000)
        self.fc21 = nn.Linear(4000, 1000)
        self.fc22 = nn.Linear(4000, 1000)
        self.fc3 = nn.Linear(1000, 4000)
        self.fc4 = nn.Linear(4000, 27648)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # DOn't take too big of a step from BCE

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = BCE + KLD

    return loss.clamp(min=-100,max=100)

def create_VAE():
    vae = VAE()
    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-5)
    return vae, vae_optimizer

def vae_update(data,vae,vae_optimizer):
    vae_optimizer.zero_grad()

    reconstructed_data,mu,logvar = vae(data)
    if((reconstructed_data != reconstructed_data).any()):
        print("reconstruction problem",(reconstructed_data != reconstructed_data).any())
        print(reconstructed_data)

    loss = loss_function(reconstructed_data,data,mu,logvar)
    print("loss:",loss)

    loss.backward()
    vae_optimizer.step()
    


if __name__ == "__main__":
    env = gym.make("CarRacing-v0")

    vae,vae_optimizer = create_VAE()
    
    # get tons of data
    while(True):
        data = []
        obs = env.reset()
        training_games = 0 
        for x in tqdm(range(1000)):
            data.append(obs)
            action = env.action_space.sample()
            obs,reward,done,info = env.step(action)

            if(done):
                obs = env.reset()
            else:
                env.render()
            x+= 1
                

        data = torch.Tensor(data).view(-1,27648)
        print((data != data).any())
        print(data)
        for x in range(1):
            vae_update(data,vae,vae_optimizer)

