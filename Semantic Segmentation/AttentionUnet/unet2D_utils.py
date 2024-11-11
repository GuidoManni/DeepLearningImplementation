import torch
import torch.nn as nn
import torch.nn.functional as F


def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]  # Get height/width of target
    tensor_size = tensor.size()[2]         # Get height/width of tensor
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class ConvBlock(nn.Module):
    '''
    In the original implementation of the UNet the Covolutional Block used in the contracting path consists of:
    - conv 3x3
    - ReLu
    - conv 3x3
    - ReLu
    - maxpool 2x2
    '''
    def __init__(self, in_channels, out_channels, maxpool = False):
        super(ConvBlock, self).__init__()

        self.maxpool = maxpool

        self.convblock_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=True), # **
            nn.ReLU(),
        )

        self.convblock_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=True),  # **
            nn.ReLU(),
        )

        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)



    # ** In the original U-Net paper and implementation, the convolutional layers do include bias terms (bias=True).
    #    However, it's worth noting that in modern deep learning practice, especially when using batch normalization,
    #    it's common to set bias=False for the convolution layers that are immediately followed by batch normalization.
    #    This is because the batch norm layer has its own learnable bias parameter, making the convolution's bias redundant.

    def forward(self, x):
        out_conv1 = self.convblock_1(x)
        out_conv2 = self.convblock_2(out_conv1)
        if self.maxpool:
            out = self.maxpool2d(out_conv2)
            return out, out_conv2
        else:
            return out_conv2


class UP(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor = 2):
        super(UP, self).__init__()

        '''
        The original U-Net paper used transposed convolution, but many modern implementations prefer 
        the following approach for upsampling with bilinear upsampling because:

        - It helps avoid checkerboard artifacts that can occur with transposed convolutions
        - It's often more computationally efficient
        - It can lead to smoother outputs
        '''

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.up(x)

class CropAndConc(nn.Module):
    def __init__(self):
        super(CropAndConc, self).__init__()

        '''
        Following the original implementation, this submodule perform two operations:
        - Crop the feature map from the contracting path
        - Concatenate the cropped feature map with the feature from the expanding path
        
        Modern U-Net implementations typically don't use cropping anymore. 
        Instead, they use padding in the convolutions to maintain spatial dimensions. 
        This makes the architecture simpler and ensures that the feature maps from the contracting path match exactly with 
        the expanding path.
        
        Benefits of using padding instead of cropping:
        - Simpler implementation
        - No loss of border information
        - Easier to predict output sizes
        - Better feature preservation at boundaries
        - More stable training in many cases
        '''
        pass

    def forward(self, tensor, target_tensor):
        cropped_tensor = crop_tensor(tensor, target_tensor)
        x = torch.cat([cropped_tensor, target_tensor], 1)
        return x


class AttentionGate(nn.Module):
    def __init__(self, input_channels, gate_channels, intermediate_channels):
        super(AttentionGate, self).__init__()

        # channel wise 2D conv for the gating signal
        self.wg = nn.Conv2d(in_channels= gate_channels, out_channels=intermediate_channels, kernel_size=1, bias=True)
        self.wl = nn.Conv2d(in_channels= input_channels, out_channels=intermediate_channels, kernel_size=1, bias=True)

        # ψ (psi) channel wise
        self.psi = nn.Sequential(nn.Conv2d(in_channels=intermediate_channels, out_channels=1, kernel_size=1, bias=True),
                               nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)




    def forward(self, gating_signal, input_features):
        '''
        - gating_signal: it is the gating signal from the decoder path at coarser scale, which contains contextual information aggregated from multiple scales
        - input_features: features from the skip connection (encoder path)
        '''

        # Downsample input_features to match gating signal size
        input_features_resized = F.interpolate(input_features, size=(gating_signal.shape[2], gating_signal.shape[3]),
                                               mode='bilinear', align_corners=False)

        # step 1: we perform two separate channel wise convolution 1x1
        wg_out = self.wg(gating_signal)
        wl_out = self.wl(input_features_resized)


        # step 2: addition of these transformation
        intermediate_output = self.relu(wg_out + wl_out)

        # step 3: psi linear transformation
        attention_coefficients = self.psi(intermediate_output)

        # step 4: Resampling if needed (if sizes don't match)
        if attention_coefficients.shape != input_features.shape:
            attention_coefficients = F.interpolate(attention_coefficients,
                                                   size=input_features.shape[2:],
                                                   mode="bilinear",
                                                   align_corners=True)
        # step 5: multiply with input features
        output = attention_coefficients * input_features
        return output










