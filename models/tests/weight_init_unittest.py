# weight_init_unittest.py

import unittest
import torch.nn as nn
import torch

from models.weight_init import xavier_init

class TestWeightInit(unittest.TestCase):
    def test_layer_norm(self):
        m = nn.LayerNorm(10)
        xavier_init(m)
        self.assertFalse(torch.all(m.weight == 0))
        self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))

    def test_linear(self):
        m = nn.Linear(10, 2)
        xavier_init(m)
        self.assertFalse(torch.all(m.weight == 0))
        self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))

    def test_conv2d(self):
        m = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        xavier_init(m)
        self.assertFalse(torch.all(m.weight == 0))
        self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))

    def test_conv3d(self):
        m = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
        xavier_init(m)
        self.assertFalse(torch.all(m.weight == 0))
        self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))

    def test_conv_transpose2d(self):
        m = nn.ConvTranspose2d(3, 64, kernel_size=3, stride=1, padding=1)
        xavier_init(m)
        self.assertFalse(torch.all(m.weight == 0))
        self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))

    def test_batchnorm2d(self):
        m = nn.BatchNorm2d(10)
        xavier_init(m)
        self.assertTrue(torch.allclose(m.weight, torch.ones_like(m.weight)))
        self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))

    def test_groupnorm(self):
        m = nn.GroupNorm(5, 10)
        xavier_init(m)
        self.assertTrue(torch.allclose(m.weight, torch.ones_like(m.weight)))
        self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))

    def test_embedding(self):
        m = nn.Embedding(10, 2, padding_idx=0)
        xavier_init(m)
        self.assertFalse(torch.all(m.weight == 0))
        self.assertTrue(torch.allclose(m.weight[0], torch.zeros_like(m.weight[0])))

    def test_transformer(self):
        m = nn.Transformer(batch_first=True)
        xavier_init(m)
        for name, param in m.named_parameters():
            if 'weight' in name:
                self.assertFalse(torch.all(param == 0))
            elif 'bias' in name:
                self.assertTrue(torch.allclose(param, torch.zeros_like(param)))

    def test_transformer_encoder(self):
        m = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True), num_layers=6)
        xavier_init(m)
        for name, param in m.named_parameters():
            if 'weight' in name:
                self.assertFalse(torch.all(param == 0))
            elif 'bias' in name:
                self.assertTrue(torch.allclose(param, torch.zeros_like(param)))

    def test_transformer_decoder(self):
        m = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True), num_layers=6)
        xavier_init(m)
        for name, param in m.named_parameters():
            if 'weight' in name:
                self.assertFalse(torch.all(param == 0))
            elif 'bias' in name:
                self.assertTrue(torch.allclose(param, torch.zeros_like(param)))

if __name__ == '__main__':
    unittest.main()