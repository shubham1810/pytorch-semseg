import torch
import torch.nn as nn
import torch.nn.functional as F

def add_conv_block(in_ch=1, out_ch=1, filter_size=3, dilate=1, last=False):
    conv_1 = nn.Conv2d(in_ch, out_ch, filter_size, padding=dilate*(1-last), dilation=dilate)
    bn_1 = nn.BatchNorm2d(out_ch)

    return [conv_1, bn_1]
    

class msdnet(nn.Module):
    def __init__(self, num_layers=25, in_channels=3, n_classes=21, out_channels=None):
        if in_channels is None:
            in_channels=3

        if out_channels is None:
            out_channels=n_classes

        super(MSDNet, self).__init__()

        self.layer_list = add_conv_block(in_ch=in_channels)
        
        current_in_channels = 1
        # Add N layers
        for i in range(num_layers):
            s = (i)%10 + 1
            self.layer_list += add_conv_block(in_ch=current_in_channels, dilate=s)
            current_in_channels += 1

        # Add final output block
        self.layer_list += add_conv_block(in_ch=current_in_channels + in_channels, out_ch=out_channels, filter_size=1, last=True)

        # Add to Module List
        self.layers = nn.ModuleList(self.layer_list)

        self.apply(self.weight_init)

    def forward(self, x):
        prev_features = []
        inp = x
        
        for i, f in enumerate(self.layers):
            # Check if last conv block
            if i==(len(self.layers) - 2):
                x = torch.cat(prev_features + [inp], 1)
            
            x = f(x)
            
            if (i+1)%2 == 0 and (not i==(len(self.layers)-1)):
                x = F.relu(x)
                # Append output into previous features
                prev_features.append(x)
                x = torch.cat(prev_features, 1)

        x = F.relu(x)
        return x