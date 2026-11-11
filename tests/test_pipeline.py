"""Offline regression checks for the reusable four-source pipeline.

Run with: python3 -m unittest discover -s tests -v
No network requests or real dataset are used.
"""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from main.conditions import SOURCE_NAMES, make_conditions
from main.config import env_optional_float, env_optional_int
from main.manifest import Sample, load_manifest, samples_from_images, validate_samples
from main.metrics import pairwise_agreement
from main.prompting import build_prompt, image_paths
from main.runner import run_experiment


ROOT = Path(__file__).resolve().parents[1]
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
)
VALID_OUTPUT = {
    "overall_risk_level": "unclear",
    "persons": [{
        "person_id": "person_0",
        "work_at_height": "unclear",
        "helmet_status": "unclear",
        "harness_presence_status": "unclear",
        "hook_attachment_status": "unclear",
        "harness_usage_status": "unclear",
        "final_risk_level": "unclear",
    }],
}


class PipelineFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.original = self.directory / "original.png"
        self.numbered = self.directory / "numbered.png"
        self.original.write_bytes(PNG)
        self.numbered.write_bytes(PNG)
        self.graph = self.directory / "graph.json"
        self.rules = self.directory / "rules.json"
        self.gold = self.directory / "gold.json"
        self.graph.write_text('{"secret_graph": "GRAPH_CONTENT"}', encoding="utf-8")
        self.rules.write_text('{"secret_rules": "RULE_CONTENT"}', encoding="utf-8")
        self.gold.write_text(json.dumps(VALID_OUTPUT), encoding="utf-8")
        self.sample = Sample(
            sample_id="sample_1",
            original_image=self.original,
            numbered_visualization=self.numbered,
            scene_graph=self.graph,
            rule_result=self.rules,
            gold=self.gold,
        )

    def write_manifest(self, rows=None):
        path = self.directory / "manifest.json"
        if rows is None:
            rows = [{
                "id": "sample_1",
                "original_image": "original.png",
                "numbered_visualization": "numbered.png",
                "scene_graph": "graph.json",
                "rule_result": "rules.json",
                "gold": "gold.json",
            }]
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def run_cli(self, *arguments):
        # -S disables installed packages: dry-run must not import an API SDK.
        environment = {
            key: value for key, value in os.environ.items()
            if not key.endswith("API_KEY") and key != "PYTHONPATH"
        }
        return subprocess.run(
            [sys.executable, "-S", "-m", "main", *map(str, arguments)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=15,
        )


class EnvironmentTests(unittest.TestCase):
    def test_explicit_none_differs_from_unset_generation_parameter(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(env_optional_float("TEST_TEMPERATURE", 0.0), 0.0)
            self.assertEqual(env_optional_int("TEST_MAX_TOKENS", 1024), 1024)
            for value in ("none", "null", "NONE"):
                os.environ["TEST_TEMPERATURE"] = value
                os.environ["TEST_MAX_TOKENS"] = value
                self.assertIsNone(env_optional_float("TEST_TEMPERATURE", 0.0))
                self.assertIsNone(env_optional_int("TEST_MAX_TOKENS", 1024))


class ConditionTests(PipelineFixture):
    def test_named_modes_have_expected_inputs(self):
        self.assertEqual(make_conditions("full"), {"full": frozenset(SOURCE_NAMES)})
        self.assertEqual(make_conditions("base"), {"base": frozenset({"original_image"})})
        conditions = make_conditions("leave_one_out")
        self.assertEqual(len(conditions), 6)
        self.assertIn("full", conditions)
        self.assertIn("base", conditions)
        self.assertEqual(sum(len(value) == 3 for value in conditions.values()), 4)

    def test_all_nonempty_subsets_are_unique_and_include_reference(self):
        conditions = make_conditions("all_subsets")
        self.assertEqual(len(conditions), 15)
        self.assertEqual(len(set(conditions.values())), 15)
        self.assertTrue(all(conditions.values()))
        self.assertEqual(conditions["full"], frozenset(SOURCE_NAMES))
        self.assertEqual(conditions["base"], frozenset({"original_image"}))

    def test_multisource_ablation_removes_content_and_images(self):
        conditions = make_conditions("ablation", drop=("scene_graph", "numbered_visualization"))
        sources = next(iter(conditions.values()))
        self.assertEqual(sources, frozenset({"original_image", "rule_result"}))
        prompt = build_prompt(self.sample, sources, "TASK")
        self.assertNotIn("GRAPH_CONTENT", prompt)
        self.assertIn("RULE_CONTENT", prompt)
        self.assertEqual(image_paths(self.sample, sources), [self.original])

    def test_base_does_not_read_removed_evidence(self):
        self.graph.unlink()
        self.rules.write_text("not valid JSON", encoding="utf-8")
        self.numbered.unlink()
        conditions = make_conditions("base")
        validate_samples([self.sample], conditions)
        prompt = build_prompt(self.sample, conditions["base"], "TASK")
        self.assertNotIn("GRAPH_CONTENT", prompt)
        self.assertNotIn("RULE_CONTENT", prompt)
        self.assertEqual(image_paths(self.sample, conditions["base"]), [self.original])

    def test_invalid_ablation_requests_are_rejected(self):
        for mode, removed in (
            ("unknown", ()),
            ("ablation", ()),
            ("ablation", SOURCE_NAMES),
            ("ablation", ("unknown",)),
            ("full", ("scene_graph",)),
        ):
            with self.subTest(mode=mode, removed=removed), self.assertRaises(ValueError):
                make_conditions(mode, drop=removed)


class ManifestTests(PipelineFixture):
    def test_paths_resolve_from_manifest_location(self):
        samples = load_manifest(self.write_manifest())
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].original_image, self.original)
        self.assertEqual(samples[0].scene_graph, self.graph)
        validate_samples(samples, make_conditions("full"))

    def test_wrapped_manifest_is_supported(self):
        path = self.write_manifest({"samples": [{"id": "a", "original_image": "original.png"}]})
        samples = load_manifest(path)
        self.assertEqual(samples[0].sample_id, "a")
        validate_samples(samples, make_conditions("base"))

    def test_duplicate_sample_ids_are_rejected(self):
        path = self.write_manifest([
            {"id": "same", "original_image": "original.png"},
            {"id": "same", "original_image": "numbered.png"},
        ])
        with self.assertRaises(ValueError):
            load_manifest(path)

    def test_missing_selected_source_is_rejected(self):
        self.graph.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            validate_samples([self.sample], make_conditions("full"))

    def test_invalid_selected_json_is_rejected(self):
        self.rules.write_text("broken JSON", encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_samples([self.sample], make_conditions("full"))

    def test_images_with_duplicate_stems_are_not_silently_merged(self):
        (self.directory / "nested").mkdir()
        (self.directory / "nested" / "original.png").write_bytes(PNG)
        try:
            samples = samples_from_images(self.directory)
        except ValueError:
            return  # Explicit rejection is also a safe policy.
        ids = [sample.sample_id for sample in samples]
        self.assertEqual(len(samples), 3)
        self.assertEqual(len(set(ids)), len(ids))


class RunnerTests(PipelineFixture):
    def test_model_failure_is_recorded_and_later_calls_continue(self):
        invoker = Mock()
        invoker.chat.side_effect = [RuntimeError("synthetic failure"), json.dumps(VALID_OUTPUT)]
        outdir = self.directory / "run"
        with redirect_stdout(io.StringIO()):
            summary = run_experiment(
                samples=[self.sample],
                conditions=make_conditions("base"),
                outdir=outdir,
                system_prompt="SYSTEM",
                task_prompt="TASK",
                repeats=2,
                invoker=invoker,
            )
        self.assertEqual(invoker.chat.call_count, 2)
        rows = [json.loads(line) for line in (outdir / "raw_results.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertIn("synthetic failure", rows[0]["error"])
        self.assertIsNone(rows[0]["parsed_output"])
        self.assertFalse(rows[0]["schema"]["valid"])
        self.assertIsNone(rows[1]["error"])
        self.assertTrue(rows[1]["schema"]["valid"])
        self.assertEqual(summary["conditions"]["base"]["json_parse_rate"], 0.5)
        self.assertEqual(summary["conditions"]["base"]["exact_match_rate"], 0.5)
        self.assertEqual(json.loads((outdir / "summary.json").read_text()), summary)

    def test_gold_is_never_included_in_model_prompt(self):
        invoker = Mock()
        invoker.chat.return_value = json.dumps(VALID_OUTPUT)
        self.gold.write_text(json.dumps({**VALID_OUTPUT, "overall_risk_level": "high"}))
        with redirect_stdout(io.StringIO()):
            run_experiment(
                samples=[self.sample], conditions=make_conditions("base"),
                outdir=self.directory / "run", system_prompt="SYSTEM",
                task_prompt="TASK", invoker=invoker,
            )
        call = invoker.chat.call_args.kwargs
        self.assertNotIn("high", call["user_prompt"])
        self.assertEqual(call["image_paths"], [self.original])

    def test_malformed_json_values_do_not_crash_evaluation(self):
        bad_outputs = (
            {"overall_risk_level": [], "persons": []},
            {"overall_risk_level": "unclear", "persons": 1},
            {"overall_risk_level": "unclear", "persons": [{"helmet_status": {}}]},
        )
        invoker = Mock()
        invoker.chat.side_effect = [json.dumps(value) for value in bad_outputs]
        with redirect_stdout(io.StringIO()):
            summary = run_experiment(
                samples=[self.sample], conditions=make_conditions("base"),
                outdir=self.directory / "run", system_prompt="SYSTEM",
                task_prompt="TASK", repeats=3, invoker=invoker,
            )
        self.assertEqual(summary["conditions"]["base"]["n_calls"], 3)
        self.assertEqual(summary["conditions"]["base"]["schema_valid_rate"], 0)

    def test_paired_delta_averages_repeats_before_pairing_images(self):
        wrong = {**VALID_OUTPUT, "overall_risk_level": "high"}
        invoker = Mock()
        # full: correct then wrong; base: wrong then correct. Each averages 0.5.
        # Using only repeat zero would incorrectly report a full-minus-base gain of 1.
        invoker.chat.side_effect = [json.dumps(value) for value in (
            VALID_OUTPUT, wrong, wrong, VALID_OUTPUT,
        )]
        conditions = {**make_conditions("full"), **make_conditions("base")}
        with redirect_stdout(io.StringIO()):
            summary = run_experiment(
                samples=[self.sample], conditions=conditions,
                outdir=self.directory / "paired", system_prompt="SYSTEM",
                task_prompt="TASK", repeats=2, invoker=invoker,
            )
        delta = summary["paired_full_minus_ablated"]["base"]["overall_risk_level"]
        self.assertEqual(delta["mean_full_minus_ablated"], 0.0)
        self.assertEqual(delta["n_pairs"], 1)
        self.assertEqual(delta["bootstrap_95_ci"], [0.0, 0.0])

    def test_two_parse_failures_are_not_perfect_agreement(self):
        agreement = pairwise_agreement([None, None])
        self.assertTrue(agreement is None or agreement == 0.0)


class CommandLineTests(PipelineFixture):
    def test_full_dry_run_needs_neither_credentials_nor_sdk(self):
        outdir = self.directory / "dry"
        result = self.run_cli(
            "--manifest", self.write_manifest(), "--outdir", outdir,
            "--env-file", self.directory / "missing.env", "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((outdir / "raw_results.jsonl").exists())

    def test_base_dry_run_accepts_raw_images_without_other_evidence(self):
        result = self.run_cli(
            "--image-dir", self.directory, "--mode", "base",
            "--outdir", self.directory / "dry", "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_existing_run_is_rejected_without_overwriting_artifacts(self):
        outdir = self.directory / "existing"
        manifest = self.write_manifest()
        first = self.run_cli("--manifest", manifest, "--outdir", outdir, "--dry-run")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        before = {path.name: path.read_bytes() for path in outdir.iterdir() if path.is_file()}
        second = self.run_cli(
            "--manifest", manifest, "--outdir", outdir, "--mode", "base", "--dry-run",
        )
        self.assertNotEqual(second.returncode, 0)
        after = {path.name: path.read_bytes() for path in outdir.iterdir() if path.is_file()}
        self.assertEqual(before, after)

    def test_invalid_inputs_fail_before_api_use(self):
        empty = self.directory / "empty"
        empty.mkdir()
        manifest = self.write_manifest()
        for extra in (
            ("--image-dir", empty, "--mode", "base"),
            ("--manifest", manifest, "--repeats", "0"),
            ("--manifest", manifest, "--mode", "ablation", "--drop", *SOURCE_NAMES),
            ("--image-dir", self.directory, "--mode", "full"),
        ):
            with self.subTest(arguments=extra):
                result = self.run_cli(*extra, "--outdir", self.directory / "bad", "--dry-run")
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_missing_later_sample_fails_preflight(self):
        path = self.write_manifest([
            {"id": "first", "original_image": "original.png"},
            {"id": "second", "original_image": "absent.png"},
        ])
        outdir = self.directory / "bad"
        result = self.run_cli("--manifest", path, "--mode", "base", "--outdir", outdir, "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((outdir / "raw_results.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
