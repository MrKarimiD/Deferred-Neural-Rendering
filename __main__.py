from _prelude import *

from renderer import *
from framebuffer import *
from random import randint
import argparse


if __name__ == '__main__':
    print("Start")
    parser = argparse.ArgumentParser(description='Make point cloud from depth images')

    parser.add_argument('--trainset',
                        default='../bake',
                        help="Train set address")

    parser.add_argument('--testset',
                        default='../test_bake',
                        help="Test set address")

    parser.add_argument('--output',
                        default='C:/Users/Mohammad Reza/Desktop/test_var',
                        help="output folder")

    parser.add_argument('--epochs', type=int, default=2000, help="number of epochs per training")
    parser.add_argument('--checkpoints', type=int, default=100, help="number of epochs per checkpoints")
    parser.add_argument('--useGPU', action='store_true')

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
    texture_features = torch.randn(1, 16, 512, 512, requires_grad=True, device=device)
    # from torch.autograd import Variable
    # texture_features = Variable(torch.randn(1, 16, 512, 512), requires_grad=True, device=device)
    renderer = Renderer(use_shading=False, device=device)
    model = Net()
    model = model.to(device)
    texture_features = texture_features.to(device)
    # # check keras-like model summary using torchsummary
    # from torchsummary import summary
    # summary(model, texture_features.shape)
    loader = FrameBufferLoader(path=args.trainset)
    testloader = FrameBufferLoader(path=args.testset)
    print("Loading data is done!")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    optimizer.add_param_group({'params': texture_features})
    criterion = nn.L1Loss()

    for i in range(args.epochs):
        print("Epoch " + str(i))
        for j, fb in enumerate(loader):

            if j >= len(loader) - 1:
                break

            optimizer.zero_grad()

            render = renderer(texture_features, fb)
            source = model(render)
            target = fb.image  # renderer(target_texture, fb)
            target = target.to(device)
            loss = criterion(target, source)
            print("Loss for entry " + str(j) + ": " + str(loss))
            loss.backward()

            if j == len(loader) - 2:
                fb = testloader[len(testloader) - 1] #[randint(0, len(testloader)-1)]
                test_render = renderer(texture_features, fb)
                test_source = model(test_render)
                test_target = fb.image
                print("Saving the output pf epoch " + str(i))
                save_image(test_source, f"{args.output}/render_{i:4d}.png")
                save_image(test_target, f"{args.output}/target_{i:4d}.png")

                if i % args.checkpoints == 0:
                    print("Saving network data")
                    torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, f"{args.output}/model_epoch_{i:4d}.pt")

            optimizer.step()
