import unittest
from torch.utils.data import Dataset, DataLoader
from abstract_data import BaseDataModule
import torch
import os

# Define a simple dataset for testing
class SimpleDataset(Dataset):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.data = torch.randn(length, 100, 4)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]


# Define a concrete subclass of BaseDataModule
class ConcreteDataModule(BaseDataModule):
    def __init__(self, length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = length
    def _create_datasets(self, stage):
        return SimpleDataset(self.length)

import math
class TestBaseDataModule(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.num_workers = 1
        self.length = 50
        self.data_module = ConcreteDataModule(
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            length=self.length,
            )


    def _base_dataloader_test(self, dataloader):
        self.assertIsInstance(dataloader, DataLoader)
        expected_length = math.ceil(self.length / self.batch_size)
        for i, batch in enumerate(dataloader):
            if i == expected_length - 1:
                self.assertEqual(batch.shape, (self.length % self.batch_size, 100, 4))
            else:
                self.assertEqual(batch.shape, (self.batch_size, 100, 4))
        self.assertTrue(i == expected_length - 1)


    def test_train_dataloader(self):
        self.data_module.setup('fit')
        dataloader = self.data_module.train_dataloader()
        self._base_dataloader_test(dataloader)
    

    def test_val_dataloader(self):
        self.data_module.setup('fit')
        dataloader = self.data_module.val_dataloader()
        self._base_dataloader_test(dataloader)


    def test_test_dataloader(self):
        self.data_module.setup('test')
        dataloader = self.data_module.test_dataloader()
        self._base_dataloader_test(dataloader)


    def test_predict_dataloader(self):
        self.data_module.setup('predict')
        dataloader = self.data_module.pred_dataloader()
        self._base_dataloader_test(dataloader)


    def test_num_workers(self):
        if self.num_workers == -1:
            self.assertEqual(self.data_module.num_workers, os.cpu_count())
        else:
            self.assertEqual(self.data_module.num_workers, self.num_workers)


if __name__ == '__main__':
    unittest.main()