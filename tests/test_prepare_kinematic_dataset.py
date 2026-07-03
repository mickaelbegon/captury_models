from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prepare_kinematic_dataset import flatten_dataset


class FlattenDatasetTests(unittest.TestCase):
    def test_flatten_dataset_creates_captury_and_motive_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "flat"
            captury = source / "Static" / "captury"
            motive = source / "Static" / "squelettes"
            captury.mkdir(parents=True)
            motive.mkdir(parents=True)

            for suffix in ("bvh", "fbx", "c3d"):
                (captury / f"P6.{suffix}").write_text(
                    f"captury-{suffix}", encoding="utf-8"
                )
            (motive / "P6_Static.c3d").write_text("motive-c3d", encoding="utf-8")
            (motive / "P6_Static.fbx").write_text("motive-fbx", encoding="utf-8")
            (motive / "P6_Static_Skeleton 001.bvh").write_text(
                "motive-bvh", encoding="utf-8"
            )
            (motive / "notes.txt").write_text("ignored", encoding="utf-8")

            rows = flatten_dataset(source, output)

            self.assertEqual(len(rows), 6)
            self.assertTrue((output / "Captury" / "Static_P6.bvh").exists())
            self.assertTrue((output / "Captury" / "Static_P6.fbx").exists())
            self.assertTrue((output / "Captury" / "Static_P6.c3d").exists())
            self.assertTrue((output / "Motive" / "P6_Static.c3d").exists())
            self.assertTrue((output / "Motive" / "P6_Static.fbx").exists())
            self.assertTrue((output / "Motive" / "P6_Static_Skeleton 001.bvh").exists())
            self.assertFalse((output / "Motive" / "notes.txt").exists())

            systems = {row["system"] for row in rows}
            kinds = {row["kind"] for row in rows}
            self.assertEqual(systems, {"Captury", "Motive"})
            self.assertEqual(kinds, {"bvh", "fbx", "c3d"})


if __name__ == "__main__":
    unittest.main()
