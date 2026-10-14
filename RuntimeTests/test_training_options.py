import random
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Runtime"))
from loom_workshop.base_models import MaterialMapNetwork
from loom_workshop.lora import attach_lora
from loom_workshop.training_options import (Lion, accumulation_windows, learning_rate_at,
                                          make_optimizer, sampling_rng, validate_options)


class TrainingOptionsTests(unittest.TestCase):
    def test_lion_allocates_momentum_only_on_first_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        optimizer = Lion([parameter], lr=0.01)
        parameter.grad = torch.ones_like(parameter)
        with patch("torch.zeros_like", wraps=torch.zeros_like) as allocate:
            for _ in range(4):
                optimizer.step()
            self.assertEqual(allocate.call_count, 1)

    def test_all_optimizers_update_and_round_trip_state(self):
        for name in ("adamw", "adam", "sgd", "lion"):
            with self.subTest(optimizer=name):
                options = validate_options({"optimizer": name})
                parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
                optimizer = make_optimizer([parameter], 0.01, options)
                parameter.square().sum().backward()
                optimizer.step()
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertTrue((parameter.abs() < 1).all())
                fresh = make_optimizer([parameter], 0.01, options)
                fresh.load_state_dict(optimizer.state_dict())
                self.assertEqual(fresh.param_groups[0]["lr"], 0.01)
                self.assertTrue(fresh.state)

    def test_lion_uses_decoupled_decay_sign_and_separate_ema(self):
        self.assertEqual(validate_options({"optimizer": "lion"})["beta2"], 0.99)
        parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
        optimizer = Lion([parameter], lr=0.1, betas=(0.9, 0.8), weight_decay=0.2)
        parameter.grad = torch.tensor([4.0, -2.0])
        optimizer.step()
        torch.testing.assert_close(parameter, torch.tensor([1.86, -2.84]))
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.tensor([0.8, -0.4]))

    def test_partial_accumulation_matches_actual_full_batches(self):
        x = torch.arange(1.0, 8.0)[:, None]
        y = 2 * x
        for name in ("adamw", "adam", "sgd", "lion"):
            with self.subTest(optimizer=name):
                options = validate_options({"optimizer": name, "weight_decay": 0})
                accumulated = torch.nn.Linear(1, 1, bias=False)
                full = torch.nn.Linear(1, 1, bias=False)
                accumulated.weight.data.fill_(0.5)
                full.load_state_dict(accumulated.state_dict())
                accum_opt = make_optimizer(accumulated.parameters(), 0.001, options)
                full_opt = make_optimizer(full.parameters(), 0.001, options)
                windows = list(accumulation_windows(list(range(7)), 2, 2))
                self.assertEqual([count for _, count in windows], [4, 3])
                self.assertEqual([len(batch) for batch in windows[-1][0]], [2, 1])
                for window, count in windows:
                    accum_opt.zero_grad()
                    for indices in window:
                        loss = torch.nn.functional.mse_loss(accumulated(x[indices]), y[indices])
                        (loss * len(indices) / count).backward()
                    accum_opt.step()
                    indices = sum(window, [])
                    full_opt.zero_grad()
                    torch.nn.functional.mse_loss(full(x[indices]), y[indices]).backward()
                    full_opt.step()
                torch.testing.assert_close(accumulated.weight, full.weight, atol=1e-7, rtol=1e-7)

    def test_schedule_reaches_endpoints_and_warmup_uses_optimizer_updates(self):
        options = validate_options({"epochs": 4, "warmup_epochs": 1, "lr_schedule": "cosine", "minimum_lr_ratio": 0.2})
        rates = [learning_rate_at(step, 8, 2, 0.01, options) for step in range(8)]
        self.assertAlmostEqual(rates[0], 0.005)
        self.assertAlmostEqual(rates[1], 0.01)
        self.assertAlmostEqual(rates[2], 0.01)
        self.assertAlmostEqual(rates[-1], 0.002)
        self.assertEqual(rates[2:], sorted(rates[2:], reverse=True))
        options["lr_schedule"] = "constant"
        self.assertAlmostEqual(learning_rate_at(7, 8, 2, 0.01, options), 0.01)
        options.update(lr_schedule="cosine", warmup_epochs=0)
        self.assertAlmostEqual(learning_rate_at(0, 1, 1, 0.01, options), 0.01)

    def test_sampling_modes_and_streams_are_independent(self):
        def values(mode, epoch=0, stream="crop"):
            return [sampling_rng(mode, 42, stream, "material", epoch, random.Random(5)).random() for _ in range(2)]
        self.assertEqual(values("per_run", 0), values("per_run", 20))
        self.assertEqual(values("per_epoch", 2), values("per_epoch", 2))
        self.assertNotEqual(values("per_epoch", 2), values("per_epoch", 3))
        self.assertNotEqual(values("per_epoch", 2), values("per_epoch", 2, "augmentation"))
        rng = random.Random(5)
        a = sampling_rng("per_sample", 42, "crop", "material", 0, rng).random()
        b = sampling_rng("per_sample", 42, "crop", "material", 0, rng).random()
        self.assertNotEqual(a, b)

    def test_parameter_preview_formula_matches_real_base_and_lora(self):
        for kind in ("depth", "roughness", "normal"):
            for width, blocks, rank in ((8, 1, 2), (24, 4, 8), (48, 6, 16)):
                model = MaterialMapNetwork(kind, width, blocks)
                outputs = 3 if kind == "normal" else 1
                base = 28 * width + blocks * (18 * width * width + 2 * width) + 9 * width * outputs + outputs
                delta = rank * (27 + width + blocks * 20 * width + 9 * width + outputs)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), base)
                attach_lora(model, rank, rank)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), delta)

    def test_invalid_training_fields_fail_before_a_run(self):
        bad = {"optimizer": "mystery", "lr_schedule": "mystery", "crop_refresh": "never",
               "augmentation_refresh": "never", "gradient_accumulation_steps": 0,
               "gradient_clip_norm": float("nan"), "beta1": 1, "beta2": -1,
               "weight_decay": -1, "minimum_lr_ratio": 2,
               "seed": -1, "warmup_epochs": 11, "flip_crops": 1}
        for key, value in bad.items():
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_options({key: value})
        with self.assertRaises(ValueError):
            validate_options({"optimizer": "sgd", "momentum": 1})
        with self.assertRaises(ValueError):
            validate_options({"weight_decay": 11})

    def test_inactive_optimizer_fields_are_normalized(self):
        options = validate_options({"optimizer": "sgd", "beta1": -1, "beta2": 1})
        self.assertEqual((options["beta1"], options["beta2"]), (0.9, 0.999))
        options = validate_options({"optimizer": "lion", "momentum": -1})
        self.assertEqual(options["momentum"], 0.9)

    def test_swift_and_runtime_boundary_contracts_match(self):
        accepted = {"epochs": 5, "warmup_epochs": 4, "learning_rate": 1,
                    "gradient_accumulation_steps": 4096, "gradient_clip_norm": 1000000,
                    "seed": 2147483647, "weight_decay": 10}
        validate_options(accepted)
        validate_options({"epochs": 1, "warmup_epochs": 0})
        for key, value in (("warmup_epochs", 5), ("learning_rate", 1.00001),
                           ("gradient_accumulation_steps", 4097), ("gradient_clip_norm", 1000001),
                           ("seed", 2147483648), ("weight_decay", 10.00001)):
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_options({**accepted, key: value})


if __name__ == "__main__":
    unittest.main()
