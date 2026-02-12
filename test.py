import os
import argparse
from tqdm import tqdm
import time
import torch
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data
from MFDNet import HPCNet as mfdnet
from skimage import img_as_ubyte

# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='RainDS Deraining using MFDNet')

parser.add_argument('--input_dir', default='D:/capstone/RainDS/RainDS_real/test_set/', type=str,
                    help='RainDS test_set directory')

parser.add_argument('--result_dir', default='./results/', type=str,
                    help='Directory for results')

parser.add_argument('--weights', default='./checkpoints/models/RainDS_MFDNet/model_best.pth', type=str,
                    help='Path to weights')

parser.add_argument('--gpus', default='0', type=str)

args = parser.parse_args()

# ---------------------------------------------------------
# GPU
# ---------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
model_restoration = mfdnet().to(device)

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['state_dict'])

model_restoration.eval()

print("===> Testing using weights:", args.weights)

# ---------------------------------------------------------
# RainDS Categories
# ---------------------------------------------------------
datasets = ['raindrop', 'rainstreak', 'rainstreak_raindrop']

for dataset in datasets:

    print(f"\nProcessing {dataset}...")

    rgb_dir_test = os.path.join(args.input_dir, dataset)

    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    all_time = 0
    count = 0

    with torch.no_grad():
        for data_test in tqdm(test_loader):

            input_ = data_test[0].to(device)
            filenames = data_test[1]

            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()

            restored = model_restoration(input_)

            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()

            cost_time = end_time - start_time
            all_time += cost_time
            count += 1

            restored = torch.clamp(restored[0], 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().numpy()

            restored_img = img_as_ubyte(restored[0])

            utils.save_img(
                os.path.join(result_dir, filenames[0] + '.png'),
                restored_img
            )

    print(f"{dataset} → Avg Time per Image: {all_time / count:.4f} sec")
