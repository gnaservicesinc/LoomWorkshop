import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Runtime"))
from loom_workshop.base_models import MaterialMapNetwork
from loom_workshop.engine import Cancellation, Cancelled, train
from loom_workshop.lora import adapter_weights, attach_lora, frozen_base_weights, fuse_lora
from loom_workshop.package import file_sha, load_base
from loom_workshop.metadata import FACTORY_KEY, WORKSHOP_KEY, read_tag, write_tag, set_raw_tag, prepare_membership


class BumpMembershipTests(unittest.TestCase):
    def test_bump_cannot_be_a_depth_target_and_ignored_metadata_stops_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diffuse, bump = root / "diffuse.png", root / "bump.png"
            for path, role in ((diffuse, "diffuse"), (bump, "bump")):
                path.write_bytes(b"original image bytes")
                write_tag(path, {"id": "bark" if role == "diffuse" else "bark-bump", "pairID": "bark-pair", "role": role})
            sample = {"id": "bark", "input": str(diffuse), "targets": {"depth": str(bump)}}
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                prepare_membership([sample])
            write_tag(bump, {"id": "bark-bump", "pairID": "bark-pair", "role": "depth"})
            prepare_membership([sample])
            set_raw_tag(bump, FACTORY_KEY, json.dumps({"id": "bark-bump", "pairingKey": "bark-pair", "role": "depth", "ignored": True}).encode())
            with self.assertRaisesRegex(ValueError, "ignored"):
                prepare_membership([sample])


class NormalConventionMembershipTests(unittest.TestCase):
    def test_current_factory_or_workshop_directx_target_is_rejected_without_an_ignore_flag(self):
        for key in (FACTORY_KEY, WORKSHOP_KEY):
            for convention in ("directx", "DirectX", "DIRECTX"):
                with self.subTest(metadata=key, convention=convention), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    diffuse, normal = root / "diffuse.png", root / "normal.png"
                    diffuse.write_bytes(b"diffuse image bytes"); normal.write_bytes(b"normal image bytes")
                    input_tag = {"id": "material", "pairID": "material-pair", "role": "diffuse"}
                    normal_tag = {"id": "material-normal", "role": "normal", "normalConvention": "opengl", "ignored": False}
                    normal_tag["pairingKey" if key == FACTORY_KEY else "pairID"] = "material-pair"
                    write_tag(diffuse, input_tag)
                    set_raw_tag(normal, key, json.dumps(normal_tag).encode())
                    sample = {"id": "material", "input": str(diffuse), "targets": {"normal": str(normal)},
                              "normal_convention": "opengl"}
                    prepare_membership([sample])
                    set_raw_tag(normal, key, json.dumps({**normal_tag, "normalConvention": convention}).encode())
                    with self.assertRaisesRegex(ValueError, "DirectX normal maps are excluded"):
                        prepare_membership([sample])
                    self.assertEqual(normal.read_bytes(), b"normal image bytes")

    def test_directx_request_is_rejected_before_new_image_tags_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diffuse, normal = root / "diffuse.png", root / "normal.png"
            diffuse.write_bytes(b"diffuse image bytes"); normal.write_bytes(b"normal image bytes")
            sample = {"id": "material", "input": str(diffuse), "targets": {"normal": str(normal)},
                      "normal_convention": "DirectX"}
            with self.assertRaisesRegex(ValueError, "DirectX normal maps are excluded"):
                prepare_membership([sample])
            self.assertIsNone(read_tag(diffuse, WORKSHOP_KEY))
            self.assertIsNone(read_tag(normal, WORKSHOP_KEY))


class CreateMLClassifierConsumerTests(unittest.TestCase):
    archive = Path("/Users/andrewsmith/Library/Application Support/Loom Factory/Validation/classifier-folder-fix/IntegrationTextureLoom.textureloom")
    native_asset = Path(__file__).resolve().parents[2] / "LoomFactory" / "Resources" / "DTTC3.mlmodel"

    def fixture(self, root):
        return InstalledBaseTrainingTests.fixture(self, root)

    def rewrite(self, base, manifest):
        manifest["files"] = [{"path": str(path.relative_to(base)), "sha256": file_sha(path)}
            for path in sorted(base.rglob("*")) if path.is_file() and path.name not in ("manifest.json", ".DS_Store")]
        (base / "manifest.json").write_text(json.dumps(manifest))

    def compile(self, base, manifest):
        import coremltools as ct
        import shutil
        original = base / manifest["classifier"]["path"]
        target = original.with_suffix(".mlmodelc")
        shutil.copytree(ct.models.utils.compile_model(str(original)), target)
        manifest["classifier"]["path"] = str(target.relative_to(base))

    @unittest.skipUnless(archive.is_file(), "Requires a genuine imported-classifier Factory export")
    def test_imported_tensor_classifier_needs_no_training_weights_and_supports_compiled_models(self):
        for compiled in (False, True):
            with self.subTest(compiled=compiled), tempfile.TemporaryDirectory() as directory:
                request, manifest, base = self.fixture(Path(directory))
                self.assertEqual(manifest["classifier"]["inputType"], "multiArray")
                self.assertNotIn("weightsPath", manifest["classifier"])
                if compiled:
                    self.compile(base, manifest)
                    self.rewrite(base, manifest)
                with contextlib.redirect_stdout(io.StringIO()):
                    plan = train(request, plan_only=True)
                self.assertTrue(plan["groups"])

    @unittest.skipUnless(archive.is_file() and native_asset.is_file(), "Requires the genuine Factory map export and trained Create ML image classifier")
    def test_native_image_classifier_routes_source_labels_to_fixed_dataset_ids(self):
        import coremltools as ct
        import shutil
        spec = ct.utils.load_spec(str(self.native_asset))
        native_labels = list(spec.pipelineClassifier.pipeline.models[-1].glmClassifier.stringClassLabels.vector)
        for compiled in (False, True):
            with self.subTest(compiled=compiled), tempfile.TemporaryDirectory() as directory:
                request, manifest, base = self.fixture(Path(directory))
                native = base / "native-classifier.mlmodel"
                shutil.copyfile(self.native_asset, native)
                original_category = manifest["categories"][0]
                labels = [{"id": f"dataset-native-{index}", "name": label, "sourceLabel": label}
                    for index, label in enumerate(native_labels)]
                manifest["categories"] = [{**original_category, "id": label["id"], "name": label["name"]} for label in labels]
                manifest["classifier"] = {"architecture": "create-ml-image-classifier-v1", "format": "coreml",
                    "path": native.name, "inputName": "image", "outputName": "target", "inputType": "image",
                    "inputWidth": spec.description.input[0].type.imageType.width,
                    "inputHeight": spec.description.input[0].type.imageType.height, "inputColorSpace": "RGB", "labels": labels}
                if compiled:
                    self.compile(base, manifest)
                self.rewrite(base, manifest)
                with contextlib.redirect_stdout(io.StringIO()):
                    plan = train(request, plan_only=True)
                self.assertEqual(len(plan["groups"]), 1)
                category_id, kind = next(iter(plan["groups"]))
                self.assertEqual(kind, "depth")
                self.assertTrue(category_id.startswith("dataset-native-"))
                self.assertNotEqual(category_id, plan["categories"][category_id]["name"])

    @unittest.skipUnless(archive.is_file(), "Requires a genuine imported-classifier Factory export")
    def test_declared_tensor_shape_must_match_native_model(self):
        with tempfile.TemporaryDirectory() as directory:
            request, manifest, base = self.fixture(Path(directory))
            manifest["classifier"]["inputShape"][-1] += 1
            self.rewrite(base, manifest)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "declared RGB tensor interface"):
                train(request, plan_only=True)


class LoRAMathTests(unittest.TestCase):
    def test_initial_adapter_is_exact_zero_and_fusion_matches_only_delta_training(self):
        torch.manual_seed(42)
        model = MaterialMapNetwork("depth", 8, 1)
        rgb = torch.rand(1, 3, 32, 40)
        expected = model(rgb).detach()
        attach_lora(model, 2, 2)
        before = frozen_base_weights(model)
        torch.testing.assert_close(model(rgb).detach(), expected, atol=0, rtol=0)
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
        optimizer.zero_grad()
        model(rgb).mean().backward()
        optimizer.step()
        self.assertTrue(any(torch.count_nonzero(value).item() for key, value in adapter_weights(model).items() if key.endswith("up.weight")))
        self.assertTrue(all(torch.equal(value, frozen_base_weights(model)[key]) for key, value in before.items()))
        self.assertTrue(all(not parameter.requires_grad for name, parameter in model.named_parameters() if ".base." in name))
        fused = fuse_lora(model)
        torch.testing.assert_close(fused(rgb), model(rgb), atol=1e-6, rtol=1e-6)
        self.assertFalse(any(".base." in key or ".up." in key or ".down." in key for key in fused.state_dict()))


class InstalledBaseTrainingTests(unittest.TestCase):
    # Opt-in fixture is a genuine output of Factory's actor integration test.
    archive = Path("/Users/andrewsmith/Library/Application Support/Loom Factory/Validation/fresh-runtime/IntegrationTextureLoom.textureloom")

    def fixture(self, root):
        base = root / "Texture Studio.app" / "Contents" / "Resources" / "TextureLoom"
        base.mkdir(parents=True)
        with zipfile.ZipFile(self.archive) as archive:
            archive.extractall(base)
        manifest = json.loads((base / "manifest.json").read_text())
        source = root / "Images"
        source.mkdir()
        rgb = np.random.default_rng(42).integers(0, 256, (48, 48, 3), dtype=np.uint8)
        target = rgb[..., 0].astype(np.uint16) * 257
        input_path, target_path = source / "brick_diffuse.png", source / "brick_height.png"
        Image.fromarray(rgb).save(input_path)
        Image.fromarray(target).save(target_path)
        request = {"schema": "loom-workshop-training-v1", "source_root": str(source),
                   "output_directory": str(source / ".loom-workshop" / "runs" / "test"),
                   "base_root": str(base), "base_id": manifest["id"], "base_version": manifest["version"],
                   "adapter_name": "Workshop test", "image_size": 32, "epochs": 1, "batch_size": 1,
                   "rank": 2, "alpha": 2, "learning_rate": 0.01, "device": "cpu",
                   "shuffle_each_epoch": False, "flip_crops": False,
                   "samples": [{"id": "brick", "input": str(input_path), "targets": {"depth": str(target_path)}}]}
        return request, manifest, base

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_real_lora_trains_only_deltas_and_exports_exact_base_package(self):
        with tempfile.TemporaryDirectory() as directory:
            request, base_manifest, base = self.fixture(Path(directory))
            original_hashes = {path: file_sha(path) for path in [request["samples"][0]["input"], request["samples"][0]["targets"]["depth"]]}
            weights_path = base / base_manifest["categories"][0]["models"][0]["weightsPath"]
            base_sha = file_sha(weights_path)
            with contextlib.redirect_stdout(io.StringIO()) as log:
                manifest = train(request)
            self.assertEqual(manifest["schema"], "texture-loom.adapter.v1")
            self.assertEqual(manifest["baseModel"], {"id": base_manifest["id"], "version": base_manifest["version"]})
            self.assertEqual(len(manifest["categories"]), 1)
            category = manifest["categories"][0]
            self.assertEqual(category["id"], base_manifest["categories"][0]["id"])
            self.assertEqual(category["version"], base_manifest["categories"][0]["version"])
            adapter = category["adapters"][0]
            self.assertEqual(adapter["baseSHA256"], base_sha)
            self.assertEqual(file_sha(weights_path), base_sha)
            package = Path(request["output_directory"]) / "adapter"
            delta = load_file(str(package / adapter["weightsPath"]))
            self.assertTrue(delta)
            self.assertTrue(all(key.endswith((".up.weight", ".down.weight")) for key in delta))
            self.assertTrue(any(torch.count_nonzero(value).item() for key, value in delta.items() if key.endswith("up.weight")))
            self.assertEqual(original_hashes, {path: file_sha(path) for path in original_hashes})
            for file in manifest["files"]:
                self.assertEqual(file_sha(package / file["path"]), file["sha256"])
            self.assertIn('"event": "classification"', log.getvalue())
            self.assertEqual(json.loads((Path(request["output_directory"]) / "result.json").read_text())["status"], "completed")

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_photos_without_targets_are_sorted_but_cannot_fake_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            request, _, _ = self.fixture(Path(directory))
            request["samples"][0]["targets"] = {}
            with contextlib.redirect_stdout(io.StringIO()):
                plan = train(request, plan_only=True)
                self.assertEqual(plan["groups"], {})
                with self.assertRaisesRegex(ValueError, "photos alone"):
                    train(request)
            self.assertFalse((Path(request["output_directory"]) / "adapter" / "manifest.json").exists())

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_changed_base_identity_and_changed_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            request, manifest, base = self.fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "base model changed"):
                train({**request, "base_version": "different"})
            weights = base / manifest["categories"][0]["models"][0]["weightsPath"]
            with weights.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                load_base(base, manifest["id"], manifest["version"])

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_cancelled_training_does_not_publish_an_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            request, _, _ = self.fixture(Path(directory))
            cancellation = Cancellation()
            def stop_after_update(event, **fields):
                if event == "progress":
                    cancellation.requested = True
            from unittest.mock import patch
            with patch("loom_workshop.engine.emit", stop_after_update), self.assertRaises(Cancelled):
                train(request, cancellation)
            output = Path(request["output_directory"])
            self.assertFalse((output / "adapter" / "manifest.json").exists())
            self.assertEqual(json.loads((output / "result.json").read_text())["status"], "cancelled")

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_non_finite_loss_cannot_publish_an_adapter(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            request, _, _ = self.fixture(Path(directory))
            def nonfinite_loss(prediction, target):
                return prediction.mean() * float("nan")
            with contextlib.redirect_stdout(io.StringIO()), patch("loom_workshop.engine.F.l1_loss", nonfinite_loss):
                with self.assertRaisesRegex(ValueError, "non-finite loss"):
                    train(request)
            output = Path(request["output_directory"])
            self.assertFalse((output / "adapter" / "manifest.json").exists())
            self.assertEqual(json.loads((output / "result.json").read_text())["updates"], 0)

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_current_exact_base_image_tag_survives_rename_and_skips_classifier(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            request, manifest, _ = self.fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                train(request, plan_only=True)
            sample = request["samples"][0]
            tag = read_tag(sample["input"], WORKSHOP_KEY)
            self.assertEqual(tag["categoryID"], manifest["categories"][0]["id"])
            renamed = Path(sample["input"]).with_name("renamed.png")
            Path(sample["input"]).rename(renamed)
            sample["input"] = str(renamed)
            sample["category_id"] = tag["categoryID"]
            with contextlib.redirect_stdout(io.StringIO()), patch("loom_workshop.engine.classifier_model", side_effect=AssertionError("A current tag must not be recategorized")):
                plan = train(request, plan_only=True)
            self.assertEqual(list(plan["groups"]), [(tag["categoryID"], "depth")])

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_reviewed_factory_tag_is_honored_and_stale_request_is_rejected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            request, manifest, _ = self.fixture(Path(directory))
            sample = request["samples"][0]
            category = manifest["categories"][0]
            for path, role in [(sample["input"], "diffuse"), (sample["targets"]["depth"], "depth")]:
                set_raw_tag(path, FACTORY_KEY, json.dumps({"id": "factory-" + role, "pairingKey": "factory-pair", "role": role,
                    "category": category["name"], "verified": True, "note": "reviewed"}).encode())
            sample["category_id"] = category["id"]
            with contextlib.redirect_stdout(io.StringIO()), patch("loom_workshop.engine.classifier_model", side_effect=AssertionError("A reviewed Factory tag must be honored")):
                train(request, plan_only=True)
            sample["category_id"] = "dataset-stale"
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "tag changed"):
                train(request, plan_only=True)

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_changed_pair_metadata_halts_export(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            request, _, _ = self.fixture(Path(directory))
            target = request["samples"][0]["targets"]["depth"]
            def change_after_update(event, **fields):
                if event == "progress":
                    tag = read_tag(target, WORKSHOP_KEY)
                    tag["pairID"] = "another-pair"
                    write_tag(target, tag)
            with patch("loom_workshop.engine.emit", change_after_update), self.assertRaisesRegex(ValueError, "metadata changed"):
                train(request)
            self.assertFalse((Path(request["output_directory"]) / "adapter" / "manifest.json").exists())

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_changed_factory_category_cannot_reuse_old_workshop_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            request, manifest, _ = self.fixture(Path(directory))
            sample = request["samples"][0]
            with contextlib.redirect_stdout(io.StringIO()):
                train(request, plan_only=True)
            sample["category_id"] = manifest["categories"][0]["id"]
            pair_id = read_tag(sample["input"], WORKSHOP_KEY)["pairID"]
            set_raw_tag(sample["input"], FACTORY_KEY, json.dumps({"id": sample["id"], "pairingKey": pair_id,
                "role": "diffuse", "category": "Removed dataset", "verified": True}).encode())
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "no model"):
                train(request, plan_only=True)

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_current_factory_tag_is_authoritative_and_clearing_reclassifies_once(self):
        from unittest.mock import patch
        from loom_workshop.engine import classifier_model
        with tempfile.TemporaryDirectory() as directory:
            request, manifest, _ = self.fixture(Path(directory))
            sample = request["samples"][0]
            factory_tag = {"id": "factory-input", "pairingKey": "factory-pair", "role": "diffuse",
                           "category": manifest["categories"][0]["name"], "verified": False}
            set_raw_tag(sample["input"], FACTORY_KEY, json.dumps(factory_tag).encode())
            with contextlib.redirect_stdout(io.StringIO()), patch("loom_workshop.engine.classifier_model", side_effect=AssertionError("A current Factory category must be honored")):
                train(request, plan_only=True)
            self.assertEqual(read_tag(sample["input"], WORKSHOP_KEY)["factoryCategory"], factory_tag["category"])
            factory_tag["category"] = None
            set_raw_tag(sample["input"], FACTORY_KEY, json.dumps(factory_tag).encode())
            with contextlib.redirect_stdout(io.StringIO()), patch("loom_workshop.engine.classifier_model", wraps=classifier_model) as load:
                train(request, plan_only=True)
                self.assertEqual(load.call_count, 1)
                train(request, plan_only=True)
                self.assertEqual(load.call_count, 1)
            self.assertIsNone(read_tag(sample["input"], WORKSHOP_KEY)["factoryCategory"])

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_target_for_another_dataset_is_absent_from_the_training_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            request, _, _ = self.fixture(Path(directory))
            sample = request["samples"][0]
            with contextlib.redirect_stdout(io.StringIO()):
                train(request, plan_only=True)
            pair_id = read_tag(sample["input"], WORKSHOP_KEY)["pairID"]
            set_raw_tag(sample["targets"]["depth"], FACTORY_KEY, json.dumps({"id": "target", "pairingKey": pair_id,
                "role": "depth", "category": "Other dataset", "verified": True}).encode())
            with contextlib.redirect_stdout(io.StringIO()):
                plan = train(request, plan_only=True)
            self.assertEqual(plan["groups"], {})

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_deleted_target_halts_export(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            request, _, _ = self.fixture(Path(directory))
            target = Path(request["samples"][0]["targets"]["depth"])
            def delete_after_update(event, **fields):
                if event == "progress":
                    target.unlink()
            with patch("loom_workshop.engine.emit", delete_after_update), self.assertRaisesRegex(ValueError, "moved or deleted"):
                train(request)
            self.assertFalse((Path(request["output_directory"]) / "adapter" / "manifest.json").exists())

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_rebuilt_base_with_same_identity_cannot_rebind_a_running_adapter(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            request, manifest, base = self.fixture(Path(directory))
            def change_package_after_update(event, **fields):
                if event == "progress":
                    manifest["name"] = "Rebuilt package"
                    (base / "manifest.json").write_text(json.dumps(manifest))
            with patch("loom_workshop.engine.emit", change_package_after_update), self.assertRaisesRegex(ValueError, "package changed"):
                train(request)
            self.assertFalse((Path(request["output_directory"]) / "adapter" / "manifest.json").exists())

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_unrelated_target_removed_during_training_does_not_affect_selected_pairs(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            request, _, _ = self.fixture(Path(directory))
            sample = request["samples"][0]
            irrelevant = Path(request["source_root"]) / "other_roughness.png"
            Image.fromarray(np.zeros((48, 48), dtype=np.uint16)).save(irrelevant)
            sample["targets"]["roughness"] = str(irrelevant)
            set_raw_tag(irrelevant, FACTORY_KEY, json.dumps({"id": "other", "pairingKey": "pair-brick",
                "role": "roughness", "category": "Other dataset", "verified": True}).encode())
            def delete_unrelated(event, **fields):
                if event == "progress":
                    irrelevant.unlink()
            with patch("loom_workshop.engine.emit", delete_unrelated):
                manifest = train(request)
            self.assertEqual(manifest["categories"][0]["adapters"][0]["map"], "depth")
            self.assertTrue((Path(request["output_directory"]) / "adapter" / "manifest.json").exists())

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_category_changed_after_classification_cannot_start_stale_training(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            request, _, _ = self.fixture(Path(directory))
            sample = request["samples"][0]
            def retag_after_classification(event, **fields):
                if event == "classification":
                    pair_id = read_tag(sample["input"], WORKSHOP_KEY)["pairID"]
                    set_raw_tag(sample["input"], FACTORY_KEY, json.dumps({"id": "changed", "pairingKey": pair_id,
                        "role": "diffuse", "category": "Other dataset", "verified": True}).encode())
            with patch("loom_workshop.engine.emit", retag_after_classification), self.assertRaisesRegex(ValueError, "no model"):
                train(request)
            self.assertFalse((Path(request["output_directory"]) / "adapter" / "manifest.json").exists())

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_plan_excludes_small_maps_and_rejects_mismatched_native_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            request, _, _ = self.fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                plan = train({**request, "image_size": 64}, plan_only=True)
                self.assertEqual(plan["groups"], {})
            Image.fromarray(np.zeros((40, 40), dtype=np.uint16)).save(request["samples"][0]["targets"]["depth"])
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "same native dimensions"):
                train(request, plan_only=True)

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_compiled_installed_classifier_is_supported(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as directory:
            request, manifest, base = self.fixture(Path(directory))
            original = base / manifest["classifier"]["path"]
            compiled = Path(ct.models.utils.compile_model(str(original)))
            import shutil
            destination = original.with_suffix(".mlmodelc")
            shutil.copytree(compiled, destination)
            manifest["classifier"]["path"] = str(destination.relative_to(base))
            manifest["files"] = [{"path": str(path.relative_to(base)), "sha256": file_sha(path)}
                for path in sorted(base.rglob("*")) if path.is_file() and path.name not in ("manifest.json", ".DS_Store")]
            (base / "manifest.json").write_text(json.dumps(manifest))
            with contextlib.redirect_stdout(io.StringIO()):
                plan = train(request, plan_only=True)
            self.assertTrue(plan["groups"])

    @unittest.skipUnless(archive.is_file(), "Requires a genuine Factory integration export fixture")
    def test_fixed_epoch_order_keeps_random_crop_positions(self):
        from unittest.mock import patch
        from loom_workshop.images import native_pair
        with tempfile.TemporaryDirectory() as directory:
            request, _, _ = self.fixture(Path(directory))
            source = Path(request["source_root"])
            samples = []
            for index in range(3):
                input_path, target_path = source / f"sample{index}_diffuse.png", source / f"sample{index}_depth.png"
                Image.fromarray(np.full((48, 48, 3), 64 + index * 32, dtype=np.uint8)).save(input_path)
                Image.fromarray(np.full((48, 48), 16000 + index * 2000, dtype=np.uint16)).save(target_path)
                samples.append({"id": str(index), "input": str(input_path), "targets": {"depth": str(target_path)}})
            request.update(samples=samples, epochs=2, seed=42, shuffle_each_epoch=False)
            seen = []
            def record(sample, *args, **kwargs):
                pair = native_pair(sample, *args, **kwargs)
                seen.append((sample["id"], pair[2]["x"], pair[2]["y"]))
                return pair
            with contextlib.redirect_stdout(io.StringIO()), patch("loom_workshop.engine.native_pair", record):
                train(request)
            self.assertEqual([row[0] for row in seen[:3]], [row[0] for row in seen[3:]])
            self.assertNotEqual([row[1:] for row in seen[:3]], [row[1:] for row in seen[3:]])
            fixed_positions = [row[1:] for row in seen]
            seen.clear()
            request.update(shuffle_each_epoch=True, output_directory=str(source / ".loom-workshop" / "runs" / "randomized"))
            with contextlib.redirect_stdout(io.StringIO()), patch("loom_workshop.engine.native_pair", record):
                train(request)
            self.assertNotEqual([row[0] for row in seen[:3]], [row[0] for row in seen[3:]])
            self.assertEqual([row[1:] for row in seen], fixed_positions)


class NativePixelTests(unittest.TestCase):
    def test_depth_feature_contrast_preserves_native16bit_values_endpoints_and_source_bytes(self):
        import random
        from loom_workshop.images import exaggerate_depth, native_pair
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, target_path = root / "diffuse.png", root / "depth.png"
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(input_path)
            codes = np.linspace(8000, 55000, 32 * 32).reshape(32, 32).astype(np.uint16)
            Image.fromarray(codes).save(target_path)
            original = file_sha(target_path)
            sample = {"id": "native", "input": str(input_path), "target": str(target_path)}
            _, plain, _ = native_pair(sample, 32, random.Random(42), "depth", False, False, False)
            _, stretched, _ = native_pair(sample, 32, random.Random(42), "depth", False, False, True)
            np.testing.assert_array_equal(plain.numpy()[0], codes.astype(np.float32) / 65535)
            self.assertEqual(float(plain.min()), float(stretched.min()))
            self.assertEqual(float(plain.max()), float(stretched.max()))
            self.assertLess(float(stretched.flatten()[256]), float(plain.flatten()[256]))
            self.assertGreater(float(stretched.flatten()[768]), float(plain.flatten()[768]))
            self.assertTrue(torch.equal(exaggerate_depth(torch.full((1, 32, 32), 0.5)), torch.full((1, 32, 32), 0.5)))
            self.assertEqual(file_sha(target_path), original)


if __name__ == "__main__":
    unittest.main()
