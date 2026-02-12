import os
from dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest


# def get_training_data(rgb_dir, img_options):
#     assert os.path.exists(rgb_dir)
#     return DataLoaderTrain(rgb_dir, img_options)


# def get_validation_data(rgb_dir, img_options):
#     assert os.path.exists(rgb_dir)
#     return DataLoaderVal(rgb_dir, img_options)


# def get_test_data(rgb_dir, img_options):
#     assert os.path.exists(rgb_dir)
#     return DataLoaderTest(rgb_dir, img_options)


def get_training_data(rgb_dir, img_options):
    if not os.path.exists(rgb_dir):
        raise RuntimeError(f"\n❌ Training directory not found: {rgb_dir}\n")
    return DataLoaderTrain(rgb_dir, img_options)


def get_validation_data(rgb_dir, img_options):
    if not os.path.exists(rgb_dir):
        raise RuntimeError(f"\n❌ Validation directory not found: {rgb_dir}\n")
    return DataLoaderVal(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    if not os.path.exists(rgb_dir):
        raise RuntimeError(f"\n❌ Test directory not found: {rgb_dir}\n")
    return DataLoaderTest(rgb_dir, img_options)

