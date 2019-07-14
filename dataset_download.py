""" Download MNIST dataset from <http://yann.lecun.com/exdb/mnist/> """

import os
import gzip
import urllib.request

def download_MNIST():
    # dataset url
    urls = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"]
    raw_folder = 'raw_data'

    # folder for download
    try:
        os.makedirs(os.path.join(raw_folder))
    except FileExistsError as e:
        pass

    # download file & unzip
    for url in urls:
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        with open(file_path.replace('.gz', ''), 'wb') as out_f:
            with gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())  # file unzip
        os.unlink(file_path)  # file remove

    print("download done.")

if __name__ == "__main__":
    download_MNIST()