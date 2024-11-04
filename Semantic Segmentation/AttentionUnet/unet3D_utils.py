import torch
import torch.nn as nn
import torch.nn.functional as F

def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2:]  # Get depth/height/width of target
    tensor_size = tensor.size()[2:]         # Get depth/height/width of tensor
    delta = [t - p for t, p in zip(tensor_size, target_size)]
    delta = [d // 2 for d in delta]
    return tensor[:, :,
                 delta[0]:tensor_size[0]-delta[0],
                 delta[1]:tensor_size[1]-delta[1],
                 delta[2]:tensor_size[2]-delta[2]]

class ConvBlock(nn.Module):
    '''
    3D version of the Convolutional Block:
    - conv 3x3x3 with padding=1
    - ReLu
    - conv 3x3x3 with padding=1
    - ReLu
    - maxpool 2x2x2
    '''
    def __init__(self, in_channels, out_channels, maxpool=False):
        super(ConvBlock, self).__init__()

        self.maxpool = maxpool

        self.convblock_1 = nn.Sequential(
            # Added padding=1 to maintain spatial dimensions
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )

        self.convblock_2 = nn.Sequential(
            # Added padding=1 to maintain spatial dimensions
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )

        self.maxpool3d = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out_conv1 = self.convblock_1(x)
        out_conv2 = self.convblock_2(out_conv1)
        if self.maxpool:
            out = self.maxpool3d(out_conv2)
            return out, out_conv2
        else:
            return out_conv2

class UP(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UP, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.up(x)

class CropAndConc(nn.Module):
    def __init__(self):
        super(CropAndConc, self).__init__()
        pass

    def forward(self, tensor, target_tensor):
        cropped_tensor = crop_tensor(tensor, target_tensor)
        x = torch.cat([cropped_tensor, target_tensor], 1)
        return x

class AttentionGate(nn.Module):
    def __init__(self, input_channels, gate_channels, intermediate_channels):
        super(AttentionGate, self).__init__()

        # channel wise 3D conv for the gating signal
        self.wg = nn.Conv3d(in_channels=gate_channels, out_channels=intermediate_channels, kernel_size=1, bias=True)
        self.wl = nn.Conv3d(in_channels=input_channels, out_channels=intermediate_channels, kernel_size=1, bias=True)

        # ψ (psi) channel wise
        self.psi = nn.Sequential(
            nn.Conv3d(in_channels=intermediate_channels, out_channels=1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gating_signal, input_features):
        '''
        - gating_signal: it is the gating signal from the decoder path at coarser scale
        - input_features: features from the skip connection (encoder path)
        '''

        # Downsample input_features to match gating signal size
        input_features_resized = F.interpolate(input_features,
                                             size=(gating_signal.shape[2],
                                                  gating_signal.shape[3],
                                                  gating_signal.shape[4]),
                                             mode='trilinear',
                                             align_corners=False)

        # step 1: we perform two separate channel wise convolution 1x1x1
        wg_out = self.wg(gating_signal)
        wl_out = self.wl(input_features_resized)

        # step 2: addition of these transformation
        intermediate_output = self.relu(wg_out + wl_out)

        # step 3: psi linear transformation
        attention_coefficients = self.psi(intermediate_output)

        # step 4: Resampling if needed (if sizes don't match)
        if attention_coefficients.shape[2:] != input_features.shape[2:]:
            attention_coefficients = F.interpolate(attention_coefficients,
                                                size=input_features.shape[2:],
                                                mode="trilinear",
                                                align_corners=True)

        # step 5: multiply with input features
        output = attention_coefficients * input_features
        return output