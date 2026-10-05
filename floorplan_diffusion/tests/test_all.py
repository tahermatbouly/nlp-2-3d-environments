"""Unit / sanity tests.  Run:  .venv/bin/python -m unittest discover -s floorplan_diffusion/tests -t . -v"""
import os
import unittest

import numpy as np
import torch
from shapely.geometry import Polygon

from floorplan_diffusion import losses
from floorplan_diffusion.config import ModelConfig, TrainConfig
from floorplan_diffusion.data import (FloorplanDataset, area_features, collate, flag_duplicates,
                                      graph_batch, load_cache)
from floorplan_diffusion.diffusion import FloorplanDiffusion
from floorplan_diffusion.generate import (generate_floorplans, sample_to_graph, to_plan_dict)
from floorplan_diffusion.geometry import preprocess_polygon, signed_area
from floorplan_diffusion.metrics import evaluate_ground_truth

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = None


def cache():
    global CACHE
    if CACHE is None:
        CACHE = load_cache(os.path.join(ROOT, "ResPlan.pkl"), os.path.join(ROOT, "split.json"),
                           os.path.join(ROOT, "cache"))
    return CACHE


def small_cfg(mode):
    return ModelConfig(mode=mode, d_model=64, n_heads=4, graph_layers=2, denoiser_layers=2)


class GeometryTests(unittest.TestCase):
    def test_pipeline_rules(self):
        # CW square with a duplicate vertex and a collinear vertex, starting mid-way
        ring = [(4, 4), (4, 4), (4, 0), (2, 0), (0, 0), (0, 4)]
        out = preprocess_polygon(Polygon(ring))
        self.assertEqual(len(out), 4)                       # dup + collinear removed
        self.assertGreater(signed_area(out), 0)             # CCW
        self.assertEqual(tuple(out[0]), (0.0, 0.0))         # canonical start (min y, min x)
        again = preprocess_polygon(Polygon(out))
        np.testing.assert_allclose(out, again)              # idempotent / deterministic

    def test_bowtie_fixed_or_rejected(self):
        out = preprocess_polygon(Polygon([(0, 0), (4, 4), (4, 0), (0, 4)]))
        if out is not None:
            self.assertTrue(Polygon(out).is_valid)

    def test_all_cached_rings_follow_rules(self):
        for s in cache()["train"][:300]:
            for i in range(len(s["room_type"])):
                r = s["coords"][i][s["vmask"][i]].astype(np.float64)
                self.assertTrue(Polygon(r).is_valid)
                self.assertGreater(signed_area(r), 0)
                cand = np.where(r[:, 1] <= r[:, 1].min() + 1e-3)[0]
                self.assertAlmostEqual(float(r[0, 0]), float(r[cand, 0].min()), places=3)
                self.assertTrue(r[0, 1] <= r[:, 1].min() + 1e-3)


class DataTests(unittest.TestCase):
    def test_no_duplicates_across_splits(self):
        c = cache()
        self.assertFalse(any(flag_duplicates(c["test"], c["train"])))
        self.assertFalse(any(flag_duplicates(c["val"], c["train"])))
        ids = [set(s["id"] for s in c[k]) for k in ("train", "val", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_normalisation_roundtrip(self):
        c = cache()
        S = c["S"]
        s = c["train"][0]
        ds = FloorplanDataset([s], S)
        b = collate([ds[0]])
        back = b["coords"][0, : len(s["room_type"])].numpy() * S
        np.testing.assert_allclose(back, s["coords"], atol=1e-4)

    def test_encoder_never_sees_coordinates(self):
        c = cache()
        m = FloorplanDiffusion(small_cfg("poly"), c["S"]).eval()
        b = collate([FloorplanDataset(c["train"][:4], c["S"])[i] for i in range(4)])
        h1, g1, _ = m.encode(b)
        b2 = dict(b); b2["coords"] = torch.randn_like(b["coords"]); b2["rect"] = torch.randn_like(b["rect"])
        h2, g2, _ = m.encode(b2)
        self.assertTrue(torch.equal(h1, h2) and torch.equal(g1, g2))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.c = cache()
        self.ds = FloorplanDataset(self.c["train"], self.c["S"])

    def test_shapes_and_mask_invariance(self):
        torch.manual_seed(0)
        for mode in ("rect", "poly"):
            m = FloorplanDiffusion(small_cfg(mode), self.c["S"]).eval()
            # pick the smallest + largest graph so padding differs
            ns = [self.ds[i]["n"] for i in range(200)]
            i_s, i_l = int(np.argmin(ns)), int(np.argmax(ns))
            single = collate([self.ds[i_s]])
            both = collate([self.ds[i_s], self.ds[i_l]])
            n = single["room_mask"].shape[1]
            x0s, toks = m.targets(single)
            x0b, tokb = m.targets(both)
            t = torch.tensor([300])
            with torch.no_grad():
                hs, gs, _ = m.encode(single)
                hb, gb, _ = m.encode(both)
                self.assertTrue(torch.allclose(hs[0, :n], hb[0, :n], atol=1e-5))     # padding-invariant encoder
                a = m.net.denoise(x0s, t, hs, gs, single["room_mask"], toks)
                b = m.net.denoise(x0b[:1, :n], t, hb[:1, :n], gb[:1], both["room_mask"][:1, :n], tokb[:1, :n])
            self.assertEqual(a.shape, x0s.shape)
            self.assertTrue(torch.allclose(a, b, atol=1e-4))

    def test_graph_encoder_permutation_equivariant(self):
        m = FloorplanDiffusion(small_cfg("rect"), self.c["S"]).eval()
        b = collate([self.ds[0]])
        n = b["room_mask"].shape[1]
        perm = torch.randperm(n)
        bp = dict(b)
        bp["room_type"] = b["room_type"][:, perm]
        bp["area"] = b["area"][:, perm]
        bp["adj"] = b["adj"][:, perm][:, :, perm]
        h, g, _ = m.encode(b)
        hp, gp, _ = m.encode(bp)
        self.assertTrue(torch.allclose(h[:, perm], hp, atol=1e-5))
        self.assertTrue(torch.allclose(g, gp, atol=1e-5))


class LossTests(unittest.TestCase):
    def test_shoelace_matches_shapely(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            ang = np.sort(rng.uniform(0, 2 * np.pi, 7))
            r = rng.uniform(5, 20, 7)
            pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
            P = torch.tensor(pts, dtype=torch.float32)[None, None]
            vm = torch.ones(1, 1, 7, dtype=torch.bool)
            self.assertAlmostEqual(float(losses.signed_area(P, vm)), Polygon(pts).area, places=2)

    def test_variable_vertex_counts_ignore_padding(self):
        sq = torch.tensor([[0., 0], [4, 0], [4, 4], [0, 4], [9, 9], [9, 9]])[None, None]
        vm = torch.tensor([[[1, 1, 1, 1, 0, 0]]], dtype=torch.bool)
        self.assertAlmostEqual(float(losses.signed_area(sq, vm)), 16.0, places=4)

    def test_pair_distance_matches_shapely(self):
        a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        b = Polygon([(13, 2), (20, 2), (20, 8), (13, 8)])
        P = torch.tensor(np.stack([np.asarray(a.exterior.coords)[:-1], np.asarray(b.exterior.coords)[:-1]]),
                         dtype=torch.float32)[None]
        vm = torch.ones(1, 2, 4, dtype=torch.bool)
        D = losses.pair_distances(P, vm)
        self.assertAlmostEqual(float(D[0, 0, 1]), a.distance(b), places=3)

    def test_overlap_loss_tracks_overlap(self):
        def two(shift):
            r = torch.tensor([[[20., 20, 20, 20], [20 + shift, 20, 20, 20]]])
            return losses.rect_to_polygon(r)
        vm = torch.ones(1, 2, 4, dtype=torch.bool)
        rm = torch.ones(1, 2, dtype=torch.bool)
        full = float(losses.overlap_loss(two(0.0), vm, rm))
        half = float(losses.overlap_loss(two(10.0), vm, rm))
        none = float(losses.overlap_loss(two(40.0), vm, rm))
        self.assertGreater(full, half)
        self.assertGreater(half, none)
        self.assertLess(none, 0.02)

    def test_gradients_finite(self):
        c = cache()
        b = collate([FloorplanDataset(c["train"], c["S"])[i] for i in range(4)])
        P = (b["coords"] * c["S"]).clone().requires_grad_(True)
        out = losses.aux_losses(P, b["vmask"], b["room_mask"], b["area"], b["adj"])
        sum(v.sum() for v in out.values()).backward()
        self.assertTrue(torch.isfinite(P.grad).all())
        for k, v in out.items():
            self.assertTrue(torch.isfinite(v).all(), k)

    def test_ground_truth_is_mostly_low_loss(self):
        c = cache()
        b = collate([FloorplanDataset(c["train"], c["S"])[i] for i in range(16)])
        P = b["coords"] * c["S"]
        out = losses.aux_losses(P, b["vmask"], b["room_mask"], b["area"], b["adj"])
        self.assertLess(float(out["area"].mean()), 0.05)
        self.assertLess(float(out["valid"].mean()), 0.2)
        self.assertLess(float(out["overlap"].mean()), 0.1)


class DiffusionTests(unittest.TestCase):
    def test_overfit_and_deterministic_sampling(self):
        torch.manual_seed(0)
        c = cache()
        ds = FloorplanDataset(c["train"][:8], c["S"])
        b = collate([ds[i] for i in range(8)])
        m = FloorplanDiffusion(small_cfg("rect"), c["S"])
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        tc = TrainConfig()
        first = None
        for it in range(150):
            out = m.training_losses(b, tc, aux_scale=1.0 if it > 50 else 0.0, aux_samples=4)
            opt.zero_grad(); out["loss"].backward(); opt.step()
            if first is None:
                first = float(out["diff"].detach())
        self.assertLess(float(out["diff"].detach()), 0.6 * first)
        m.eval()
        g1 = torch.Generator().manual_seed(7); g2 = torch.Generator().manual_seed(7)
        s1 = m.sample(b, steps=5, generator=g1)["x"]; s2 = m.sample(b, steps=5, generator=g2)["x"]
        self.assertTrue(torch.equal(s1, s2))
        s3 = m.sample(b, steps=5, generator=torch.Generator().manual_seed(8))["x"]
        self.assertFalse(torch.equal(s1, s3))                 # different noise -> different layout

    def test_poly_pipeline_end_to_end(self):
        c = cache()
        m = FloorplanDiffusion(small_cfg("poly"), c["S"])     # untrained: only checks plumbing
        G = [sample_to_graph(s) for s in c["val"][:2]]
        res = generate_floorplans(m, G, K=3, steps=4, device="cpu")
        self.assertEqual(len(res), 2)
        for cands in res:
            self.assertEqual(len(cands), 3)
            plan = to_plan_dict(cands[0])
            self.assertIn("graph", plan)
            for _, d in cands[0].graph.nodes(data=True):
                self.assertIn("geometry", d)
            for u, v, d in cands[0].graph.edges(data=True):
                self.assertIn("type", d)

    def test_ground_truth_csr_reference(self):
        r = evaluate_ground_truth(cache()["val"][:200])
        self.assertGreater(r["csr"], 0.5)                     # checker is not absurdly strict
        self.assertEqual(r["invalid_polygon"], 0.0)


if __name__ == "__main__":
    unittest.main()
