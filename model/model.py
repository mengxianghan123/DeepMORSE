import torch.nn as nn
import torch.nn.functional as F

class MORSE(nn.Module):
    def __init__(self,input_dim, hidden_dim, z_dim):
        super().__init__()
        self.img_feature = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, z_dim),
                                         )
        self.text_feature = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, z_dim),
                                         )

    def forward(self, img, text):
        img_z = self.img_feature(img)
        text_z = self.text_feature(text)
        img_z = F.normalize(img_z, 2)
        text_z = F.normalize(text_z, 2)
        return img_z, text_z
    
