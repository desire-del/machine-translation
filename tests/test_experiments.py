"""Directory-based checkpoint loading tests, runnable with unittest."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from machine_translation import build_model, load_checkpoint, load_config, load_experiment
from machine_translation.config import PROJECT_ROOT
from machine_translation.tokenization import build_tokenizer, save_tokenizer
from machine_translation.training import resolve_checkpoint


class ExperimentLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.run = self.directory / "renamed_run"
        (self.run / "checkpoints").mkdir(parents=True)

    def create_run(self, model_type="seq2seq"):
        content = load_config().snapshot()
        content["name"] = "fixture"
        content["data"]["max_sequence_length"] = 6
        content["model"] = {
            "type": model_type,
            "embedding_dim": 8,
            "hidden_dim": 12,
            "num_layers": 1,
            "dropout": 0.0,
        }
        if model_type == "seq2seq_att":
            content["model"]["att_dim"] = 10
        for role in ("source", "target"):
            content["tokenizer"][f"{role}_artifact_path"] = str(
                self.directory / f"{role}.json"
            )
        self.config_path = self.run / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(content), encoding="utf-8")
        self.config = load_config(self.config_path)
        tokenizer = build_tokenizer(
            ["hello world", "bonjour le monde"],
            vocab_size=32,
            special_tokens=self.config.tokenizer.special_tokens,
            model_type="bpe",
            min_frequency=1,
        )
        save_tokenizer(tokenizer, self.config.tokenizer.source_artifact_path)
        save_tokenizer(tokenizer, self.config.tokenizer.target_artifact_path)
        self.model = build_model(
            tokenizer, tokenizer, self.config.tokenizer.special_tokens, self.config.model
        )
        for kind, epoch in (("best", 2), ("last", 3)):
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config.snapshot(),
                    "epoch": epoch,
                },
                self.run / "checkpoints" / f"{kind}.pt",
            )

    def test_load_both_architectures_and_exact_weights(self):
        for model_type in ("seq2seq", "seq2seq_att"):
            with self.subTest(model_type=model_type):
                self.create_run(model_type)
                loaded = load_experiment(self.run)
                self.assertEqual(loaded.config.model.type, model_type)
                self.assertEqual(loaded.epoch, 2)
                self.assertFalse(loaded.model.training)
                self.assertEqual(next(loaded.model.parameters()).device.type, "cpu")
                for key, value in self.model.state_dict().items():
                    torch.testing.assert_close(loaded.model.state_dict()[key], value)
                for role in ("source", "target"):
                    (self.directory / f"{role}.json").unlink()

    def test_directory_and_file_inputs(self):
        self.create_run()
        for path in (self.run, self.run / "checkpoints", self.run / "checkpoints/last.pt"):
            with self.subTest(path=path):
                self.assertEqual(load_experiment(path, checkpoint="last").epoch, 3)

    def test_config_beside_checkpoint(self):
        self.create_run()
        (self.run / "checkpoints/best.pt").rename(self.run / "best.pt")
        self.assertEqual(load_experiment(self.run).epoch, 2)

    def test_missing_best_does_not_fall_back_to_last(self):
        self.create_run()
        (self.run / "checkpoints/best.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "No best.pt"):
            load_experiment(self.run)

    def test_missing_config(self):
        self.create_run()
        self.config_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "config"):
            load_experiment(self.run)

    def test_missing_tokenizer(self):
        self.create_run()
        self.config.tokenizer.source_artifact_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Tokenizer"):
            load_experiment(self.run)

    def test_config_checkpoint_mismatch(self):
        self.create_run()
        content = yaml.safe_load(self.config_path.read_text())
        content["model"]["hidden_dim"] = 16
        self.config_path.write_text(yaml.safe_dump(content), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ in 'model'"):
            load_experiment(self.run)

    def test_ambiguous_directory(self):
        self.create_run()
        torch.save({}, self.run / "best.pt")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            resolve_checkpoint(self.run)

    def test_low_level_loader_accepts_directory(self):
        self.create_run()
        saved = load_checkpoint(
            self.run / "checkpoints", self.model, "cpu", config=self.config,
            checkpoint="last",
        )
        self.assertEqual(saved["epoch"], 3)

    def test_cli_uses_saved_config_for_both_architectures(self):
        for model_type in ("seq2seq", "seq2seq_att"):
            with self.subTest(model_type=model_type):
                self.create_run(model_type)
                result = subprocess.run(
                    [
                        sys.executable, str(PROJECT_ROOT / "scripts/translate.py"),
                        "hello", "--run-dir", str(self.run),
                        "--checkpoint-kind", "last", "--device", "cpu", "--beam-size", "1",
                    ],
                    capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for role in ("source", "target"):
                    (self.directory / f"{role}.json").unlink()

    def test_cli_rejects_conflicting_seed(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/translate.py"),
             "hello", "--run-dir", str(self.run), "--seed", "42"],
            capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined", result.stderr)

    def test_cli_checkpoint_and_legacy_config_modes(self):
        self.create_run("seq2seq_att")
        for arguments in (
            ["--checkpoint", str(self.run / "checkpoints/best.pt")],
            ["--checkpoint", str(self.run / "checkpoints")],
            ["--config", str(self.config_path),
             "--checkpoint", str(self.run / "checkpoints/best.pt")],
        ):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, str(PROJECT_ROOT / "scripts/translate.py"),
                     "hello", "--device", "cpu", "--beam-size", "1", *arguments],
                    capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
