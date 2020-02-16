import pandas
import torch
from torch.utils.data import Dataset,DataLoader

class Astronomy(Dataset):
    '''dataset class for our astronomy data
    input samples are actually in the form of pandas.DataFrame'''
    def __init__(self, samples=None, type='train'):
        self.__type = type  # train_set  or test_set
        if self.__type == 'test':
            self._samples = samples
        else:
            self._samples = samples[samples['answer'] != 'answer'] # drop 'answer' that is not labeled
            # self.resample()     #you can do some re-sample since star is far more than galaxy and qso
        self.cls2label = {"star":0, "galaxy":1, "qso":2}  # map classes to labels

    def __getitem__(self, idx):
        ''' 2600 features first
        with id following, optionally then answer'''
        self._sample = self._samples.iloc[idx]
        self._id = self._sample['id']
        self._data = torch.Tensor(self._sample[:2600])

        if self.__type == 'test':
            return self._data, self._id
        else:
            self._label = self.cls2label[self._sample['answer']]
            return self._data, self._label, self._id

    def __len__(self):
        return len(self._samples)


if __name__ == '__main__':
    path = './data/'
    train_path, test_path = path + 'new_columns_trains_sets.csv', path + 'val_sets_v1.csv'

    train_reader, test_reader = pandas.read_csv(train_path, chunksize=256), pandas.read_csv(test_path, chunksize=4096)
    train_, test_ = next(train_reader), next(test_reader)
    train_set, test_set = Astronomy(next(train_reader)), Astronomy(next(test_reader), 'test')

    train_loader, test_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=8), \
                                DataLoader(test_set, batch_size=4096, shuffle=True, num_workers=8)
    print('loader prepared')
    for data, labels, _ids in train_loader:
        print(data.shape, labels, type(labels), labels.data)
    for data, ids in test_loader:
        print(data.shape, len(ids))


