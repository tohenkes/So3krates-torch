import os
import shutil
import tempfile
import unittest

import torch
from torch_geometric.data import Data

from so3krates_torch.data.datasets import (
    BaseAtomicDataset,
    DiskCache,
    XYZDataset,
)

EXAMPLE_XYZ = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "examples", "training", "md17_ethanol_small.xyz",
)


class TestDiskCache(unittest.TestCase):
    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_save_load_roundtrip(self):
        cache = DiskCache(
            cache_dir=self.cache_dir,
            cutoff=5.0,
        )
        # Create a minimal AtomicData-like object via torch.save
        data = Data(x=torch.randn(3, 4), edge_index=torch.zeros(2, 0, dtype=torch.long))
        cache.save(0, data)
        loaded = cache.load(0)
        self.assertIsNotNone(loaded)
        self.assertTrue(torch.allclose(data.x, loaded.x))

    def test_load_missing_returns_none(self):
        cache = DiskCache(
            cache_dir=self.cache_dir,
            cutoff=5.0,
        )
        self.assertIsNone(cache.load(999))

    def test_invalidation_on_cutoff_change(self):
        cache = DiskCache(
            cache_dir=self.cache_dir,
            cutoff=5.0,
        )
        data = Data(x=torch.randn(3, 4), edge_index=torch.zeros(2, 0, dtype=torch.long))
        cache.save(0, data)
        self.assertIsNotNone(cache.load(0))

        # Re-create cache with different cutoff — should invalidate
        cache2 = DiskCache(
            cache_dir=self.cache_dir,
            cutoff=6.0,
        )
        self.assertIsNone(cache2.load(0))


@unittest.skipUnless(
    os.path.isfile(EXAMPLE_XYZ),
    f"Example XYZ file not found at {EXAMPLE_XYZ}",
)
class TestXYZDataset(unittest.TestCase):
    def test_len_and_getitem(self):
        ds = XYZDataset(
            file_path=EXAMPLE_XYZ,
            cutoff=5.0,
        )
        self.assertGreater(len(ds), 0)
        item = ds[0]
        # Should be an AtomicData (subclass of Data)
        self.assertIsInstance(item, Data)
        self.assertIn("positions", item)
        self.assertIn("edge_index", item)
        self.assertIn("atomic_numbers", item)

    def test_lazy_mode(self):
        ds = XYZDataset(
            file_path=EXAMPLE_XYZ,
            cutoff=5.0,
            lazy=True,
        )
        self.assertGreater(len(ds), 0)
        item = ds[0]
        self.assertIsInstance(item, Data)
        self.assertIn("positions", item)

    def test_iter_configs(self):
        ds = XYZDataset(
            file_path=EXAMPLE_XYZ,
            cutoff=5.0,
        )
        configs = list(ds.iter_configs())
        self.assertEqual(len(configs), len(ds))
        # Configs should have positions and atomic_numbers
        self.assertIsNotNone(configs[0].positions)
        self.assertIsNotNone(configs[0].atomic_numbers)

    def test_cached_mode(self):
        cache_dir = tempfile.mkdtemp()
        try:
            cache = DiskCache(
                cache_dir=cache_dir,
                cutoff=5.0,
                source_file=EXAMPLE_XYZ,
            )
            ds = XYZDataset(
                file_path=EXAMPLE_XYZ,
                cutoff=5.0,
                cache=cache,
            )
            # First access builds graph and caches
            item0 = ds[0]
            self.assertIsNotNone(item0)
            # Second access should hit cache
            item0_cached = ds[0]
            self.assertTrue(
                torch.allclose(item0.positions, item0_cached.positions)
            )
            # Verify cache file exists
            self.assertTrue(
                os.path.isfile(os.path.join(cache_dir, "00000000.pt"))
            )
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_dataloader_integration(self):
        """Verify dataset works with PyG DataLoader."""
        from mace.tools import torch_geometric

        ds = XYZDataset(
            file_path=EXAMPLE_XYZ,
            cutoff=5.0,
        )
        loader = torch_geometric.dataloader.DataLoader(
            dataset=ds,
            batch_size=2,
            shuffle=False,
        )
        batch = next(iter(loader))
        self.assertIsInstance(batch, Data)
        # Batch should have combined nodes from 2 graphs
        self.assertGreater(batch.positions.shape[0], 0)
        self.assertIsNotNone(batch.batch)


if __name__ == "__main__":
    unittest.main()
