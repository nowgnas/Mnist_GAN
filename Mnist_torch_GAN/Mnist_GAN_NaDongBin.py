import torch.nn as nn

latent_dim = 100


# G 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                # 배치 정규화 수행
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # activation LeakyReLU
            return layers

        # 생성자 모델은 연속적인 여러개 블록을 가진다
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 28 * 28),  # Mnist 데이터 생성
            nn.Tanh()
        )

    def forward(self, z):
        # noise vector z
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)  # 배치사이즈, 채널, 높이, 너비
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forword(self, img):
        flattened = img.view(img.size(0), -1)  # 벡터로 나열
        output = self.model(flattened)  # 모델에 집어넣음

        return output
