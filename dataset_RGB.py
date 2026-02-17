import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random

def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        self.rgb_dir = rgb_dir
        self.ps = img_options['patch_size']

        self.samples = []

        scene_folders = sorted(os.listdir(rgb_dir))

        for scene in scene_folders:
            scene_path = os.path.join(rgb_dir, scene)

            if not os.path.isdir(scene_path):
                continue

            files = sorted(os.listdir(scene_path))

            rainy_images = [f for f in files if '-R-' in f and is_image_file(f)]
            clean_images = [f for f in files if '-C-' in f and is_image_file(f)]

            for r_img in rainy_images:
                c_img = r_img.replace('-R-', '-C-')

                if c_img in clean_images:
                    rain_path = os.path.join(scene_path, r_img)
                    clean_path = os.path.join(scene_path, c_img)

                    self.samples.append((rain_path, clean_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"\n❌ No GT-RAIN training images found in {rgb_dir}\n")

        print(f"✅ GT-RAIN Training pairs loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):

        ps = self.ps

        rain_path, gt_path = self.samples[index]

        rain_img = Image.open(rain_path).convert('RGB')
        gt_img   = Image.open(gt_path).convert('RGB')

        rain_img = TF.to_tensor(rain_img)
        gt_img   = TF.to_tensor(gt_img)
 
        _, h, w = gt_img.shape

        pad_h = max(ps - h, 0)
        pad_w = max(ps - w, 0)

        if pad_h > 0 or pad_w > 0:
            rain_img = TF.pad(rain_img, (0, 0, pad_w, pad_h))
            gt_img   = TF.pad(gt_img,   (0, 0, pad_w, pad_h))

        _, h, w = gt_img.shape

        rr = random.randint(0, h - ps)
        cc = random.randint(0, w - ps)

        rain_img = rain_img[:, rr:rr+ps, cc:cc+ps]
        gt_img   = gt_img[:, rr:rr+ps,   cc:cc+ps]

        filename = os.path.splitext(os.path.basename(gt_path))[0]

        return gt_img, rain_img, filename
    


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderVal, self).__init__()

        self.rgb_dir = rgb_dir
        self.samples = []

        scene_folders = sorted(os.listdir(rgb_dir))

        for scene in scene_folders:
            scene_path = os.path.join(rgb_dir, scene)

            if not os.path.isdir(scene_path):
                continue

            files = sorted(os.listdir(scene_path))

            rainy_images = [f for f in files if '-R-' in f and is_image_file(f)]
            clean_images = [f for f in files if '-C-' in f and is_image_file(f)]

            for r_img in rainy_images:
                c_img = r_img.replace('-R-', '-C-')

                if c_img in clean_images:
                    rain_path = os.path.join(scene_path, r_img)
                    clean_path = os.path.join(scene_path, c_img)

                    self.samples.append((rain_path, clean_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"\n❌ No GT-RAIN validation images found in {rgb_dir}\n")

        print(f"✅ GT-RAIN Validation pairs loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rain_path, clean_path = self.samples[index]

        rain_img = TF.to_tensor(Image.open(rain_path).convert('RGB'))
        clean_img = TF.to_tensor(Image.open(clean_path).convert('RGB'))

        filename = os.path.splitext(os.path.basename(rain_path))[0]

        return clean_img, rain_img, filename
