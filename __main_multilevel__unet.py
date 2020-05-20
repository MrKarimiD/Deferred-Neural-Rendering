from _prelude import *

from renderer import *
from framebuffer import *
from random import randint
import argparse
from unet import UNet


if __name__ == '__main__':
    print("Start")
    parser = argparse.ArgumentParser(description='Make point cloud from depth images')

    parser.add_argument('--trainset',
                        default='C:/Users/Mohammad Reza/Desktop/remote/Neural Renderer/synthetic_ds/room_simple/output/train',
                        help="Train set address")

    parser.add_argument('--testset',
                        default='C:/Users/Mohammad Reza/Desktop/remote/Neural Renderer/synthetic_ds/room_simple/output/test',
                        help="Test set address")

    parser.add_argument('--output',
                        default='C:/Users/Mohammad Reza/Desktop/test',
                        help="output folder")

    parser.add_argument('--epochs', type=int, default=2, help="number of epochs per training")
    parser.add_argument('--checkpoints', type=int, default=100, help="number of epochs per checkpoints")
    parser.add_argument('--useGPU', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.useGPU else 'cpu')
    print("Code is running on " + str(device))
    texture_features1 = torch.randn(1, 16, 64, 64, requires_grad=True, device=device)
    texture_features2 = torch.randn(1, 16, 128, 128, requires_grad=True, device=device)
    texture_features3 = torch.randn(1, 16, 256, 256, requires_grad=True, device=device)
    texture_features4 = torch.randn(1, 16, 512, 512, requires_grad=True, device=device)
    # from torch.autograd import Variable
    # texture_features = Variable(torch.randn(1, 16, 512, 512), requires_grad=True, device=device)
    renderer = Renderer(use_shading=False, device=device)
    model = UNet(n_channels=16, n_classes=3, bilinear=False)
    model = model.to(device)
    texture_features1 = texture_features1.to(device)
    texture_features2 = texture_features2.to(device)
    texture_features3 = texture_features3.to(device)
    texture_features4 = texture_features4.to(device)
    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, (16, 512, 1024))
    loader = FrameBufferLoader(path=args.trainset)
    testloader = FrameBufferLoader(path=args.testset)
    print("Loading data is done!")
    # optimizer = torch.optim.Adam(params=[source_texture], lr=0.1)
    # all_params = [texture_features, model.parameters()]
    # all_params = list(texture_features) + list(model.parameters())
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    optimizer.add_param_group({'params': texture_features1})
    optimizer.add_param_group({'params': texture_features2})
    optimizer.add_param_group({'params': texture_features3})
    optimizer.add_param_group({'params': texture_features4})
    criterion = nn.L1Loss()

    for i in range(args.epochs + 1):
        print("Epoch " + str(i))
        for j, fb in enumerate(loader):

            if j >= len(loader) - 1:
                break

            optimizer.zero_grad()

            render_4 = renderer(texture_features4, fb)
            render_3 = renderer(texture_features3, fb)
            render_3_upsampled = F.interpolate(render_3, size=(512, 1024), mode='bilinear')
            render_2 = renderer(texture_features2, fb)
            render_2_upsampled = F.interpolate(render_2, size=(512, 1024), mode='bilinear')
            render_1 = renderer(texture_features1, fb)
            render_1_upsampled = F.interpolate(render_1, size=(512, 1024), mode='bilinear')
            render = render_1_upsampled + render_2_upsampled + render_3_upsampled + render_4
            source = model(render)
            target = fb.image  # renderer(target_texture, fb)
            target = target.to(device)
            loss = criterion(target, source) + 0.005 * torch.norm(texture_features4, 2) + 0.002 * torch.norm(texture_features3, 2) + 0.001 * torch.norm(texture_features2, 2) + torch.norm(texture_features1, 2)
            print("Loss for entry " + str(j) + ": " + str(loss))
            loss.backward()

            if j == len(loader) - 2:
                fb = testloader[len(testloader) - 1] #[randint(0, len(testloader)-1)]
                render_4 = renderer(texture_features4, fb)
                render_3 = renderer(texture_features3, fb)
                render_3_upsampled = F.interpolate(render_3, size=(512, 1024), mode='bilinear')
                render_2 = renderer(texture_features2, fb)
                render_2_upsampled = F.interpolate(render_2, size=(512, 1024), mode='bilinear')
                render_1 = renderer(texture_features1, fb)
                render_1_upsampled = F.interpolate(render_1, size=(512, 1024), mode='bilinear')
                test_render = render_1_upsampled + render_2_upsampled + render_3_upsampled + render_4
                test_source = model(test_render)
                test_target = fb.image
                print("Saving the output pf epoch " + str(i))
                save_image(test_source, f"{args.output}/render_{i:04}.png")
                save_image(test_target, f"{args.output}/target_{i:04}.png")

                if i % args.checkpoints == 0:
                    print("Saving network data")
                    torch.save(texture_features1, f"{args.output}/texture_l1_epoch_{i:04}.pt")
                    torch.save(texture_features2, f"{args.output}/texture_l2_epoch_{i:04}.pt")
                    torch.save(texture_features3, f"{args.output}/texture_l3_epoch_{i:04}.pt")
                    torch.save(texture_features4, f"{args.output}/texture_l4_epoch_{i:04}.pt")
                    torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, f"{args.output}/model_epoch_{i:04}.pt")

            optimizer.step()
