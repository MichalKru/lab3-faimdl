import os
import shutil
import urllib.request
import zipfile


def download_dataset():
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = "tiny-imagenet-200.zip"

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('dataset')

    with open('dataset/tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
        print(f'Downloading dataset...')
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

            shutil.copyfile(f'dataset/tiny-imagenet/tiny-imagenet-200/val/images/{fn}',
                            f'dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

    shutil.rmtree('dataset/tiny-imagenet/tiny-imagenet-200/val/images')

if __name__ == "__main__":
    download_dataset()