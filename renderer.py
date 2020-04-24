from __future__ import annotations

from _prelude import *

from framebuffer import *


class Renderer(nn.Module):
    def __init__(self, device: torch.device, use_shading: bool = False):
        super().__init__()
        self.use_shading = use_shading
        self.device = device

    def forward(self, texture: torch.Tensor, fb: FrameBuffer) -> torch.Tensor:
        alpha_mask = fb.uv[:, :, 2] # * Alpha map is in the UV render blue channel

        uvs_to_grid_coords_data = uvs_to_grid_coords(fb.uv)
        uvs_to_grid_coords_data = uvs_to_grid_coords_data.to(self.device)
        color = F.grid_sample(texture, uvs_to_grid_coords_data, mode='bilinear', align_corners=False)
        color = color.to(self.device)

        if self.use_shading:
            return (color * fb.diffuse + fb.specular) * alpha_mask
        else:
            return color


def uvs_to_grid_coords(uvs: torch.Tensor):
    # * Looks at grid_sample doc for file format
    # * Note that this also unsqueezes the tensor
    return ((uvs[:, :, :2] - 0.5) * torch.tensor([2.0, -2.0])).unsqueeze(0)


class TestRenderer(unittest.TestCase):
    def test_render_uv(self):
        os.makedirs("output/test", exist_ok=True)
        loader = FrameBufferLoader(path="bake")
        renderer = Renderer()

        fake_image = torch.randn(1, 3, 512, 512)

        for i, fb in enumerate(loader):
            print(i)
            render = renderer(fake_image, fb)
            save_image(render, f"output/test/render_{i}.png")
            if i >= 4:
                break

    def test_uv_to_grid(self):
        conversions = [
            ([0.0, 1.0, 1.0], [-1.0, -1.0]),
            ([0.0, 0.0, 1.0], [-1.0, 1.0]),
            ([1.0, 1.0, 1.0], [1.0, -1.0]),
            ([1.0, 0.0, 1.0], [1.0, 1.0])
        ]

        for (source, target) in conversions:
            source = torch.tensor(source).unsqueeze(0).unsqueeze(0)
            target = torch.tensor(target).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.assertTrue(uvs_to_grid_coords(source).equal(target))


if __name__ == '__main__':
    unittest.main()