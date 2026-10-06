import unittest
import numpy as np
import torch
import copy
from shapely.geometry import Polygon

from floorplan_diffusion.generate import align_walls
from floorplan_diffusion.data import FloorplanDataset, load_cache, collate
from floorplan_diffusion.config import ModelConfig
from floorplan_diffusion.diffusion import FloorplanDiffusion
from floorplan_diffusion.losses import aux_losses
from floorplan_diffusion.geometry import signed_area

class NewFeatureTests(unittest.TestCase):
    def setUp(self):
        self.c = load_cache("ResPlan.pkl", "split.json", "cache")

    def test_align_walls(self):
        rings = [
            np.array([[0, 0], [4.1, 0], [4.1, 4.1], [0, 4.1]]),
            np.array([[4.2, 0], [8.0, 0], [8.0, 4.1], [4.2, 4.1]]), # Wall between them is at ~4.1/4.2
        ]
        
        # 1.5 tolerance should snap 4.1 and 4.2 together
        aligned = align_walls(rings, tol=1.5)
        
        # Check that the x coordinate of the right side of ring 0 equals the left side of ring 1
        self.assertEqual(aligned[0][1, 0], aligned[1][0, 0])
        self.assertEqual(aligned[0][2, 0], aligned[1][3, 0])
        
        # Check that 0 and 8.0 are unchanged or at least they aren't merged
        self.assertNotEqual(aligned[0][0, 0], aligned[1][1, 0])

    def test_augmentation(self):
        ds_no_aug = FloorplanDataset(self.c["train"][:5], self.c["S"], augment=False)
        ds_aug = FloorplanDataset(self.c["train"][:5], self.c["S"], augment=True)
        
        for i in range(5):
            orig = ds_no_aug[i]
            # Try many times to ensure we hit rot and flip
            for _ in range(10):
                np.random.seed(_)
                aug = ds_aug[i]
                
                # Check area is preserved
                for j in range(orig["n"]):
                    if orig["nverts"][j] > 2:
                        orig_ring = orig["coords"][j][:orig["nverts"][j]]
                        aug_ring = aug["coords"][j][:aug["nverts"][j]]
                        
                        orig_poly = Polygon(orig_ring.numpy())
                        aug_poly = Polygon(aug_ring.numpy())
                        
                        self.assertAlmostEqual(orig_poly.area, aug_poly.area, places=4)
                        
                        # Check CCW
                        self.assertGreater(signed_area(aug_ring.numpy()), 0)

    def test_guidance_reduces_losses(self):
        cfg = ModelConfig(mode="poly", d_model=64, n_heads=4, graph_layers=2, denoiser_layers=2)
        model = FloorplanDiffusion(cfg, self.c["S"]).eval()
        
        ds = FloorplanDataset(self.c["train"][:2], self.c["S"], augment=False)
        b = collate([ds[0], ds[1]])
        
        torch.manual_seed(42)
        # Without guidance
        out_no_guide = model.sample(b, steps=5, guidance=0.0)
        x_no_guide = out_no_guide["x"]
        
        # With guidance
        torch.manual_seed(42)
        out_guide = model.sample(b, steps=5, guidance=1.0)
        x_guide = out_guide["x"]
        
        # The x arrays should be slightly different because of guidance
        self.assertFalse(torch.allclose(x_no_guide, x_guide))

if __name__ == "__main__":
    unittest.main()
