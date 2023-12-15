import torch
import unittest
from models.abstract_models import BaseModel

class BoringBaseModel(BaseModel):
    def __init__(self, in_size : int, out_size : int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = torch.nn.Linear(in_size, out_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestBaseModel(unittest.TestCase):
    def test_weight_init(self):
        model = BoringBaseModel(10,10)

        # Check if all modules have been initialized
        for module in model.modules():
            for param in module.parameters():
                self.assertIsNotNone(param.data)

        # check that the bias is initialized to 0
        self.assertTrue(
            torch.all(model.fc.bias == 0)
            )

        # check that the weights are not all 0
        self.assertTrue(
            torch.any(model.fc.weight != 0)
            )


if __name__ == '__main__':
    unittest.main()