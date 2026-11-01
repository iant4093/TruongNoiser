"""Small numerical checks for the conversion-only downsample layout rewrite."""
import copy
import unittest

import torch
from cgnet_reference import prepare_conversion_only_downsamples


class CGNetDownsampleLoweringTests(unittest.TestCase):
    def holder(self):
        torch.manual_seed(291026)
        holder = torch.nn.Module()
        holder.downs = torch.nn.ModuleList([torch.nn.Conv2d(3, 6, 2, 2) for _ in range(4)])
        return holder.eval()

    def testEveryChannelAndSpatialOffsetKeepsItsLearnedCoefficient(self):
        model = self.holder()
        reference = copy.deepcopy(model)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        self.assertEqual(prepare_conversion_only_downsamples(model), 4)
        for size in (4, 16, 32):
            value = torch.randn(1, 3, size, size)
            for original, lowered in zip(reference.downs, model.downs):
                torch.testing.assert_close(original(value), lowered(value), rtol=1e-6, atol=1e-6)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def testInvalidGeometryIsRejectedBeforeAnyModuleIsMutated(self):
        model = self.holder()
        model.downs[2] = torch.nn.Conv2d(3, 6, 3, 2)
        forwards = [module.forward for module in model.downs]
        with self.assertRaises(RuntimeError):
            prepare_conversion_only_downsamples(model)
        self.assertEqual([module.forward for module in model.downs], forwards)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
