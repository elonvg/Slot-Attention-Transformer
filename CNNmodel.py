import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class CNNencoder(nn.Module):
    def __init__(self, img_c=1, encoder_features=[32, 32, "pool", 64, "pool"], out_channels=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential()
        
        curr_channels = img_c
        for item in encoder_features:
            if item == "pool":
                self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                feature_dim = item
                self.encoder.append(nn.Conv2d(curr_channels, feature_dim, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False))
                self.encoder.append(nn.BatchNorm2d(feature_dim))
                self.encoder.append(nn.ReLU())
                curr_channels = feature_dim
        
        # Final layer
        self.encoder.append(nn.Conv2d(curr_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False))
        self.encoder.append(nn.BatchNorm2d(out_channels))
        self.encoder.append(nn.ReLU())

    def forward(self, x):
        # print(f"og size: {x.shape}")
        x = self.encoder(x)
        
        return x


class CNNdecoder(nn.Module):
    def __init__(self, in_channels=34, out_size = (64, 64), img_c=1, decoder_features=[128, "up", 64, 32, "up", 32]):
        super().__init__()

        self.decoder = nn.Sequential()
        self.final_layer = nn.Sequential(
            # nn.Upsample(size=out_size, mode="bilinear"),
            nn.Conv2d(decoder_features[-1], img_c, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(img_c),
            nn.ReLU()
            # nn.Sigmoid() # Linear activation
        )
        curr_channels = in_channels


        for item in decoder_features:
            if item == "up":
                self.decoder.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            else:
                feature_dim = item
                self.decoder.append(nn.Conv2d(curr_channels, feature_dim, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False))
                self.decoder.append(nn.BatchNorm2d(feature_dim))
                self.decoder.append(nn.ReLU())
                curr_channels = feature_dim
        
    def forward(self, x):
        x = self.decoder(x)
        x = self.final_layer(x)
        # Shape: (batch_size * num_slots, c_img, height, width)
        return x

                                    




if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128)
    encoder = CNNencoder()
    y = encoder(x)

    # in_channels = 130
    # slots = torch.randn(8, in_channels, 32, 32)
    # decoder = CNNdecoder(in_channels=in_channels)
    # y = decoder(slots)

    # print(y.shape)

