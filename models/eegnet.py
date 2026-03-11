import torch 
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, num_classes=4, num_channels=22, num_timepoints=256, F1=8, D=2, F2=16, kernel_length=64
    , dropout=0.5):
        super(EEGNet, self).__init__()


        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(F1)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1*D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        # Block 3
        self.block3 = nn.Sequential(
            # Depthwise part
            nn.Conv2d(F1*D, F1*D, kernel_size=(1, 16), padding='same', groups=F1*D, bias=False),
            # Pointwise part
            nn.Conv2d(F1*D, F2, kernel_size=(1, 1), bias=False),
            # Batch normalization
            nn.BatchNorm2d(F2),
            # Activation    
            nn.ELU(),
            # Pooling
            nn.AvgPool2d((1, 8)),
            # Dropout
            nn.Dropout(dropout/2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (num_timepoints // 32), num_classes)
        )

    def apply_weight_constraints(self):
        with torch.no_grad():
            # Spatial conv (block2[0]): max norm = 1.0
            w = self.block2[0].weight
            norms = w.reshape(w.shape[0], -1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            self.block2[0].weight.copy_(
                w * (norms.clamp(max=1.0) / (norms + 1e-8))
            )

            # Classifier linear layer: max norm = 0.25
            w = self.classifier[1].weight
            norms = w.norm(dim=1, keepdim=True)
            self.classifier[1].weight.copy_(
                w * (norms.clamp(max=0.25) / (norms + 1e-8))
            )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = EEGNet()
    dummy_input = torch.zeros(32, 1, 22, 256) 
    output = model(dummy_input)
    print("Output shape:", output.shape) 
    
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params) 

