import h5py, torch, cv2
import numpy as np
from PIL import Image
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler


class CacheDataset(torch.utils.data.Dataset):
    r"""A wrapper dataset to cache pre-processed data. It can cache data on RAM or a secondary memory.
    .. note::
        Since converting image into spike-wave can be time consuming, we recommend to wrap your dataset into a :attr:`CacheDataset`
        object.
    Args:
        dataset (torch.utils.data.Dataset): The reference dataset object.
        cache_address (str, optional): The location of cache in the secondary memory. Use :attr:`None` to cache on RAM. Default: None
    """
    def __init__(self, dataset, cache_address=None):
        self.dataset = dataset
        self.cache_address = cache_address
        self.cache = [None] * len(self.dataset)

    def __getitem__(self, index):
        if self.cache[index] is None:
            #cache it
            sample, target = self.dataset[index]
            if self.cache_address is None:
                self.cache[index] = sample, target
            else:
                save_path = os.path.join(self.cache_address, str(index))
                torch.save(sample, save_path + ".cd")
                torch.save(target, save_path + ".cl")
                self.cache[index] = save_path
        else:
            if self.cache_address is None:
                sample, target = self.cache[index]
            else:
                sample = torch.load(self.cache[index] + ".cd")
                target = torch.load(self.cache[index] + ".cl")
        return sample, target

    def reset_cache(self):
        r"""Clears the cached data. It is useful when you want to change a pre-processing parameter during
        the training process.
        """
        if self.cache_address is not None:
            for add in self.cache:
                os.remove(add + ".cd")
                os.remove(add + ".cl")
        self.cache = [None] * len(self)

    def __len__(self):
        return len(self.dataset)

class H5Dataset(torch.utils.data.Dataset):
    
    def __init__(self, f_path, length=None, transform=None):
        super().__init__()
        self.f_path = f_path
        self.length = length
        self.data, self.target = self._load_data(self.f_path)
        self.shape = self.data.shape
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        data = Image.fromarray(data.astype('uint8'), 'L')
        data = cv2.equalizeHist(np.array(data))
        target = self.target[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, int(target)
    
    def _load_data(self, file_path):
        f = h5py.File(file_path, 'r')
        
        if self.length:
            data = f['x'][:self.length]
            target = f['y'][:self.length]
        else:
            data = f['x'][:][1:]
            target = f['y'][1:]
        h, w = data.shape[1], data.shape[2]
        print('Original dataset shape {}'.format(Counter(target)))
        ros = RandomUnderSampler()
        data = np.reshape(data, (data.shape[0], h*w))
        data, target = ros.fit_sample(data, target)
        data = np.reshape(data, (data.shape[0], h, w))
        print('Resampled dataset shape {} \n'.format(Counter(target)))
        print('Dataset size : {}'.format(data.shape))
        return data, target