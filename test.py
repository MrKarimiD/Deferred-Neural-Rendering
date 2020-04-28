from _prelude import *

from renderer import *
from framebuffer import *
from random import randint
import argparse


if __name__ == '__main__':
    print("Start")
    parser = argparse.ArgumentParser(description='Make point cloud from depth images')

    parser.add_argument('--testset',
                        default='C:/Users/Mohammad Reza/Desktop/remote/synthetic_ds/vase-obj/Vase_bake_test_hard',
                        help="Test set address")

    parser.add_argument('--output',
                        default='C:/Users/Mohammad Reza/Desktop/test_var',
                        help="output folder")

    parser.add_argument('--model',
                        default='C:/Users/Mohammad Reza/Desktop/outputs/final_vase/model_epoch_1900.pt',
                        help="Model checkpoint")

    parser.add_argument('--texture',
                        default='C:/Users/Mohammad Reza/Desktop/outputs/final_vase/texture_epoch_1900.pt',
                        help="Model checkpoint")

    args = parser.parse_args()

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(16, 64, kernel_size=4, stride=2)
            self.conv1_in = nn.InstanceNorm2d(64)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
            self.conv2_in = nn.InstanceNorm2d(128)

            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
            self.conv3_in = nn.InstanceNorm2d(256)

            self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2)
            self.conv4_in = nn.InstanceNorm2d(512)

            self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2)
            self.conv5_in = nn.InstanceNorm2d(512)

            self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2)
            self.deconv1_in = nn.InstanceNorm2d(512)

            self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2)
            self.deconv2_in = nn.InstanceNorm2d(512)

            self.deconv3 = nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2)
            self.deconv3_in = nn.InstanceNorm2d(256)

            self.deconv4 = nn.ConvTranspose2d(384, 128, kernel_size=4, stride=2, output_padding=1)
            self.deconv4_in = nn.InstanceNorm2d(128)

            self.deconv5 = nn.ConvTranspose2d(192, 3, kernel_size=4, stride=2)
            self.deconv5_in = nn.InstanceNorm2d(3)

        def forward(self, x):
            # Encoder
            x = F.leaky_relu(self.conv1_in(self.conv1(x)), negative_slope=0.2)
            residual1 = x
            x = F.leaky_relu(self.conv2_in(self.conv2(x)), negative_slope=0.2)
            residual2 = x
            x = F.leaky_relu(self.conv3_in(self.conv3(x)), negative_slope=0.2)
            residual3 = x
            x = F.leaky_relu(self.conv4_in(self.conv4(x)), negative_slope=0.2)
            residual4 = x
            x = F.leaky_relu(self.conv5_in(self.conv5(x)), negative_slope=0.2)
            # Decoder
            x = F.leaky_relu(self.deconv1_in(self.deconv1(x)), negative_slope=0.2)
            x = F.leaky_relu(self.deconv2_in(self.deconv2(torch.cat([x, residual4], dim=1))), negative_slope=0.2)
            x = F.leaky_relu(self.deconv3_in(self.deconv3(torch.cat([x, residual3], dim=1))), negative_slope=0.2)
            x = F.leaky_relu(self.deconv4_in(self.deconv4(torch.cat([x, residual2], dim=1))), negative_slope=0.2)
            x = F.tanh(self.deconv5_in(self.deconv5(torch.cat([x, residual1], dim=1))))

            return x

    device = torch.device('cuda' if torch.cuda.is_available() and args.useGPU else 'cpu')
    print("Code is running on " + str(device))
    texture_features = torch.load(args.texture, map_location=device)  #randn(1, 16, 512, 512, requires_grad=True, device=device)
    # from torch.autograd import Variable
    # texture_features = Variable(torch.randn(1, 16, 512, 512), requires_grad=True, device=device)
    renderer = Renderer(use_shading=False, device=device)
    model = Net()
    model = model.to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    texture_features = texture_features.to(device)
    # # check keras-like model summary using torchsummary
    # from torchsummary import summary
    # summary(model, texture_features.shape)
    testloader = FrameBufferLoader(path=args.testset)
    print("Loading data is done!")

    for j, fb in enumerate(testloader):

        if j >= len(testloader) - 1:
            break

        render = renderer(texture_features, fb)
        source = model(render)
        target = fb.image  # renderer(target_texture, fb)

        save_image(source, f"{args.output}/render_{j:04}.png")
        save_image(target, f"{args.output}/target_{j:04}.png")