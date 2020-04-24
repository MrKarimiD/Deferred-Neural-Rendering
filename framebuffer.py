from __future__ import annotations

from _prelude import *

@dataclass
class FrameBuffer:
    uv: np.array
    diffuse: np.array
    normal: np.array
    specular: np.array
    image: np.array

    def __post_init__(self):
        assert self.uv.shape == self.diffuse.shape \
            and self.diffuse.shape == self.specular.shape \
            and self.specular.shape == self.normal.shape \
            and self.normal.shape == self.image.shape
        
        # * The color buffer are in C,W,H format, the info buffers are in W,H,C format
        self.diffuse = self.diffuse.permute((2, 0, 1))
        self.specular = self.specular.permute((2, 0, 1))
        self.image = self.image.permute((2, 0, 1))
        
        assert self.diffuse.shape[0] == 3
        assert self.specular.shape[0] == 3
        assert self.image.shape[0] == 3
        assert self.normal.shape[2] == 3
        assert self.uv.shape[2] == 3
        

@dataclass
class FrameBufferLoader(torch.utils.data.Dataset):
    path: str

    subdirs = list(FrameBuffer.__annotations__.keys())
    
    def __post_init__(self):
        def dir_len(d): return len(os.listdir(self.path + '/' + d))
        assert all([dir_len(d) == dir_len(self.subdirs[0]) for d in self.subdirs])

    def __len__(self):
        return len(os.listdir(f"{self.path}/{self.subdirs[0]}"))

    def __getitem__(self, i: int):
        i += 1 # Blender counts from 1
        # buffers = { subdir:open_16bit_png(f"{self.path}/{subdir}/{subdir}_{i:04}.png")
        #                 for subdir in self.subdirs }
        buffers = {subdir: open_16bit_png(f"{self.path}/{subdir}/{subdir}{i:04}.png")
                   for subdir in self.subdirs}
        return FrameBuffer(**buffers)


def open_16bit_png(path: str) -> torch.Tensor:
    with open(path, 'rb') as file:
        reader = png.Reader(file=file)
        (h, w, data, info) = reader.read()

        assert info['bitdepth'] == 16
        assert info['planes'] == 3
        assert info['alpha'] == False
        assert info['greyscale'] == False

        image = np.vstack(list(map(np.uint16, data))) # TODO this is slow
        image = image.reshape((w, h, 3))
        image = torch.tensor(image.astype(np.float64) / 2**16).to(dtype=torch.float32)
        
        # * Note that the alpha in the blue channel of the uvs
        assert len(image.shape) == 3
        assert image.min() >= 0.0
        assert image.max() <= 1.0
        assert image.shape[0] == image.shape[1]
        
        return image


if __name__ == '__main__':
    loader = FrameBufferLoader(path="../bake")

    for i, fb in enumerate(loader):

        os.makedirs("output/test", exist_ok=True)
        save_image(fb.uv.permute((2, 0, 1)).unsqueeze(0), f"output/test/uv_{i}.png")
        save_image(fb.diffuse, f"output/test/diffuse_{i}.png")
        save_image(fb.specular, f"output/test/specular_{i}.png")
        if i >= 4:
            break