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

    val_dir = 'dataset/tiny-imagenet-200/val'
    with open(os.path.join(val_dir, 'val_annotations.txt')) as f:
        print('Reorganizing validation set')
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            shutil.copyfile(
                os.path.join(val_dir, 'images', fn),
                os.path.join(val_dir, cls, fn)
            )

    shutil.rmtree(os.path.join(val_dir, 'images'))

if __name__ == "__main__":
    download_dataset()