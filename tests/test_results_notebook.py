"""Verify multi-experiment notebook loading without large checkpoints or CUDA."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import torch

from machine_translation.config import PROJECT_ROOT


class ResultsNotebookTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(
            (PROJECT_ROOT / "notebooks/02_seq2seq_results.ipynb").read_text()
        )
        self.cells = {
            cell["id"]: "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        }
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.loaded = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(type="seq2seq", hidden_dim=12, att_dim=None),
                seed=42,
                training=SimpleNamespace(teacher_forcing_ratio=0.5),
                tokenizer=object(),
                data=SimpleNamespace(
                    test_path=self.root / "test.parquet",
                    source_language="en", target_language="fr",
                ),
            ),
            model=torch.nn.Linear(2, 2),
            source_tokenizer=object(), target_tokenizer=object(),
            epoch=1, checkpoint_path=self.root / "best.pt",
        )
        self.namespace = {
            "PROJECT_ROOT": self.root, "json": json, "pd": pd, "torch": torch,
            "display": Mock(), "load_experiment": Mock(return_value=self.loaded),
        }

    def execute(self, cell_id):
        exec(self.cells[cell_id], self.namespace)

    def test_all_code_cells_compile(self):
        for cell_id, source in self.cells.items():
            with self.subTest(cell_id=cell_id):
                compile(source, cell_id, "exec")

    def test_no_experiments(self):
        for cell_id in ("checkpoint-path", "load-results", "history-table"):
            self.execute(cell_id)
        self.assertEqual(self.namespace["experiments"], {})
        self.assertTrue(self.namespace["summary"].empty)

    def test_all_seeds_load_and_incomplete_run_is_reported(self):
        directories = []
        for name in ("fixture/seed_42", "fixture/seed_43", "incomplete/seed_42"):
            directory = self.root / "experiments" / name
            directory.mkdir(parents=True)
            (directory / "config.yaml").touch()
            (directory / "history.json").write_text(json.dumps({
                "train_loss": [3.0], "val_loss": [2.0],
                "val_perplexity": [7.4], "epoch_time": [1.0],
            }))
            directories.append(directory)

        def load(directory, device, checkpoint):
            self.assertEqual(device, "cpu")
            self.assertEqual(checkpoint, "best")
            if directory == directories[-1]:
                raise FileNotFoundError("No best.pt")
            return self.loaded

        self.namespace["load_experiment"].side_effect = load
        for cell_id in ("checkpoint-path", "load-results", "history-table", "49a6c7e4"):
            self.execute(cell_id)
        self.assertEqual(len(self.namespace["experiments"]), 2)
        self.assertEqual(len(self.namespace["summary"]), 2)
        self.assertEqual(len(self.namespace["loading_errors"]), 1)
        self.assertEqual(self.namespace["history_table"].index.names, ["experiment", "epoch"])

        self.namespace["translate"] = Mock(return_value="bonjour")
        with patch.object(pd, "read_parquet", return_value=pd.DataFrame({
            "en": ["hello"], "fr": ["bonjour"],
        })) as read:
            self.execute("dbe495a3")
        self.assertEqual(len(self.namespace["translations"]), 2)
        self.assertEqual(self.namespace["translate"].call_count, 2)
        read.assert_called_once()
        self.assertEqual(next(self.loaded.model.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
