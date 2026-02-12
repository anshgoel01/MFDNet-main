import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random


def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))


# ==========================================================
# ===================== TRAIN LOADER ======================
# ==========================================================

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        self.rgb_dir = os.path.join(rgb_dir, "train_set")
        self.gt_dir = os.path.join(self.rgb_dir, "gt")

        self.input_dirs = [
            os.path.join(self.rgb_dir, "raindrop"),
            os.path.join(self.rgb_dir, "rainstreak"),
            os.path.join(self.rgb_dir, "rainstreak_raindrop")
        ]

        self.ps = img_options['patch_size']

        self.inp_filenames = []
        self.tar_filenames = []

        for input_dir in self.input_dirs:

            if not os.path.exists(input_dir):
                print(f"⚠ WARNING: Missing folder {input_dir}")
                continue

            files = sorted(os.listdir(input_dir))

            for fname in files:
                if is_image_file(fname):

                    inp_path = os.path.join(input_dir, fname)
                    tar_path = os.path.join(self.gt_dir, fname)

                    if os.path.exists(tar_path):
                        self.inp_filenames.append(inp_path)
                        self.tar_filenames.append(tar_path)

        self.sizex = len(self.tar_filenames)

        if self.sizex == 0:
            raise RuntimeError(f"\n❌ No training images found in {self.rgb_dir}\n")

        print(f"✅ Training samples loaded: {self.sizex}")

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):

        ps = self.ps

        inp_path = self.inp_filenames[index]
        tar_path = self.tar_filenames[index]

        try:
            inp_img = Image.open(inp_path).convert('RGB')
            tar_img = Image.open(tar_path).convert('RGB')
        except Exception as e:
            print(f"❌ Corrupt image: {inp_path}")
            raise e

        w, h = tar_img.size

        padw = max(ps - w, 0)
        padh = max(ps - h, 0)

        if padw > 0 or padh > 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)

        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        aug = random.randint(0, 7)

        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)

        filename = os.path.splitext(os.path.basename(tar_path))[0]

        return tar_img, inp_img, filename


# ==========================================================
# ===================== VALIDATION ========================
# ==========================================================

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderVal, self).__init__()

        self.rgb_dir = os.path.join(rgb_dir, "test_set")
        self.gt_dir = os.path.join(self.rgb_dir, "gt")

        self.input_dirs = [
            os.path.join(self.rgb_dir, "raindrop"),
            os.path.join(self.rgb_dir, "rainstreak"),
            os.path.join(self.rgb_dir, "rainstreak_raindrop")
        ]

        self.ps = img_options['patch_size']

        self.inp_filenames = []
        self.tar_filenames = []

        for input_dir in self.input_dirs:

            if not os.path.exists(input_dir):
                print(f"⚠ WARNING: Missing folder {input_dir}")
                continue

            files = sorted(os.listdir(input_dir))

            for fname in files:
                if is_image_file(fname):

                    inp_path = os.path.join(input_dir, fname)
                    tar_path = os.path.join(self.gt_dir, fname)

                    if os.path.exists(tar_path):
                        self.inp_filenames.append(inp_path)
                        self.tar_filenames.append(tar_path)

        self.sizex = len(self.tar_filenames)

        if self.sizex == 0:
            raise RuntimeError(f"\n❌ No validation images found in {self.rgb_dir}\n")

        print(f"✅ Validation samples loaded: {self.sizex}")

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):

        inp_path = self.inp_filenames[index]
        tar_path = self.tar_filenames[index]

        inp_img = TF.to_tensor(Image.open(inp_path).convert('RGB'))
        tar_img = TF.to_tensor(Image.open(tar_path).convert('RGB'))

        filename = os.path.splitext(os.path.basename(tar_path))[0]

        return tar_img, inp_img, filename


# ==========================================================
# ===================== TEST ONLY =========================
# ==========================================================

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options=None):
        super(DataLoaderTest, self).__init__()

        files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [
            os.path.join(inp_dir, x)
            for x in files if is_image_file(x)
        ]

        if len(self.inp_filenames) == 0:
            raise RuntimeError(f"\n❌ No test images found in {inp_dir}\n")

        print(f"✅ Test samples loaded: {len(self.inp_filenames)}")

    def __len__(self):
        return len(self.inp_filenames)

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.basename(path_inp))[0]

        inp = TF.to_tensor(Image.open(path_inp).convert('RGB'))

        return inp, filename
