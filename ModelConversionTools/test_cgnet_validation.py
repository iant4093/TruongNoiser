"""Lightweight tests for compiled tensor declarations and the Python Float16 bridge.

No torch/coremltools import, checkpoint load, conversion or prediction occurs.
Run with the retained conversion Python: python -m unittest discover
    -s ModelConversionTools -p test_cgnet_validation.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from verify_cgnet_parity import (
    compiled_identity, raw_refusal_failure, raw_summary,
    validate_compiled_tensor_contract, validate_prediction_bridge_dtype,
)


class CompiledTensorContractTests(unittest.TestCase):
    def metadata(self, size=256):
        return [{
            'inputSchema': [{'name': 'input_tile', 'dataType': 'Float16',
                             'shape': json.dumps([1, 3, size, size])}],
            'outputSchema': [{'name': 'output_tile', 'dataType': 'Float16',
                              'shape': json.dumps([1, 3, size, size])}],
        }]

    def validate(self, payload, size=256):
        with tempfile.TemporaryDirectory(prefix='CGNet-schema-') as temporary:
            directory = Path(temporary)
            (directory / 'metadata.json').write_text(json.dumps(payload))
            validate_compiled_tensor_contract(directory, size)

    def test_both_actual_fixed_shapes_accept_declared_half_ports(self):
        for size in (256, 512):
            with self.subTest(size=size):
                self.validate(self.metadata(size), size)

    def test_each_float32_declared_port_is_rejected_even_if_values_could_be_half(self):
        for schema in ('inputSchema', 'outputSchema'):
            with self.subTest(schema=schema):
                payload = self.metadata()
                payload[0][schema][0]['dataType'] = 'Float32'
                with self.assertRaisesRegex(RuntimeError, 'tensor contract'):
                    self.validate(payload)

    def test_wrong_names_on_either_port_are_rejected(self):
        for schema in ('inputSchema', 'outputSchema'):
            with self.subTest(schema=schema):
                payload = self.metadata()
                payload[0][schema][0]['name'] = 'other_tensor'
                with self.assertRaises(RuntimeError):
                    self.validate(payload)

    def test_batch_channel_spatial_and_rank_mismatches_are_rejected(self):
        invalid_shapes = ([2, 3, 256, 256], [1, 4, 256, 256],
                          [1, 3, 256, 512], [1, 3, 512, 512], [3, 256, 256])
        for schema in ('inputSchema', 'outputSchema'):
            for shape in invalid_shapes:
                with self.subTest(schema=schema, shape=shape):
                    payload = self.metadata()
                    payload[0][schema][0]['shape'] = json.dumps(shape)
                    with self.assertRaises(RuntimeError):
                        self.validate(payload)

    def test_missing_or_multiple_ports_cannot_hide_a_second_contract(self):
        for schema in ('inputSchema', 'outputSchema'):
            for count in (0, 2):
                with self.subTest(schema=schema, count=count):
                    payload = self.metadata()
                    payload[0][schema] = payload[0][schema] * count
                    with self.assertRaises(RuntimeError):
                        self.validate(payload)

    def test_missing_or_multiple_model_descriptions_are_rejected(self):
        base = self.metadata()
        for payload in ([], base + copy.deepcopy(base), {'inputSchema': []}):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(RuntimeError, 'one compiled'):
                    self.validate(payload)

    def test_missing_shape_or_element_type_is_rejected(self):
        for schema in ('inputSchema', 'outputSchema'):
            for key in ('shape', 'dataType'):
                with self.subTest(schema=schema, key=key):
                    payload = self.metadata()
                    del payload[0][schema][0][key]
                    with self.assertRaises(RuntimeError):
                        self.validate(payload)


class PredictionBridgeTests(unittest.TestCase):
    def test_each_ordinary_reference_orientation_keeps_its_invalid_prediction_failure(self):
        safe = raw_summary(np.asarray([.5], dtype=np.float16), np)
        unsafe = raw_summary(np.asarray([np.nan], dtype=np.float16), np)
        # A different unsafe-reference orientation cannot absorb this failure in a case-wide refusal.
        failures = [raw_refusal_failure(unsafe, unsafe), raw_refusal_failure(safe, unsafe)]
        self.assertEqual(failures, [None, 'invalid-converted-prediction'])
        self.assertEqual(raw_refusal_failure(unsafe, safe), 'masked-unsafe-reference')
        self.assertIsNone(raw_refusal_failure(safe, safe))

    def test_compiled_identity_rejects_symbolic_link_roots_and_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); asset = root / 'model.mlmodelc'; asset.mkdir()
            (asset / 'weight.bin').write_bytes(b'weights')
            linked = root / 'linked.mlmodelc'; linked.symlink_to(asset, target_is_directory=True)
            with self.assertRaises(RuntimeError): compiled_identity(linked)
            (asset / 'linked.bin').symlink_to(asset / 'weight.bin')
            with self.assertRaises(RuntimeError): compiled_identity(asset)

    def test_native_half_values_are_accepted_without_mutation(self):
        values = np.asarray([0, -.25, .1, .5, 1, 1.25], dtype=np.float16)
        original = values.copy()
        validate_prediction_bridge_dtype(values, np)
        np.testing.assert_array_equal(values, original)
        self.assertEqual(values.dtype, np.float16)

    def test_exact_promoted_half_values_include_subnormals_and_half_extremes(self):
        half = np.asarray([0, -0., -.25, .5, 1.25, 2 ** -24,
                           -(2 ** -24), np.finfo(np.float16).max], dtype=np.float16)
        promoted = half.astype(np.float32)
        original_bits = promoted.view(np.uint32).copy()
        validate_prediction_bridge_dtype(promoted, np)
        np.testing.assert_array_equal(promoted.view(np.uint32), original_bits)
        self.assertEqual(promoted.dtype, np.float32)

    def test_arbitrary_float32_is_refused_instead_of_silently_quantized(self):
        for value in (.1, .50001, 1 + 2 ** -20, 2 ** -25, 65520.):
            with self.subTest(value=value):
                array = np.asarray([.5, value], dtype=np.float32)
                original = array.copy()
                with self.assertRaisesRegex(RuntimeError, 'Float16 lattice'):
                    validate_prediction_bridge_dtype(array, np)
                np.testing.assert_array_equal(array, original)

    def test_integer_bool_and_double_bridges_are_not_accepted_as_half_ports(self):
        for dtype in (np.float64, np.int16, np.int32, np.bool_):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(RuntimeError, 'bridge type'):
                    validate_prediction_bridge_dtype(np.asarray([0, 1], dtype=dtype), np)

    def test_nonfinite_values_remain_visible_for_raw_refusal_after_bridge_validation(self):
        for dtype in (np.float16, np.float32):
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(dtype=dtype, bad=bad):
                    values = np.asarray([.5, bad], dtype=dtype)
                    validate_prediction_bridge_dtype(values, np)
                    state = raw_summary(values, np)
                    self.assertFalse(state['finite'])
                    self.assertTrue(state['requiresProductionRefusal'])
                    self.assertEqual(state['minFinite'], .5)
                    self.assertEqual(state['maxFinite'], .5)

    def test_all_nonfinite_tensor_is_not_a_fidelity_pass_or_a_json_nan(self):
        values = np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)
        validate_prediction_bridge_dtype(values, np)
        state = raw_summary(values, np)
        self.assertTrue(state['requiresProductionRefusal'])
        self.assertIsNone(state['minFinite'])
        self.assertIsNone(state['maxFinite'])
        json.dumps(state, allow_nan=False)

    def test_finite_half_values_still_require_raw_domain_validation(self):
        values = np.asarray([-.5, 1.5], dtype=np.float16).astype(np.float32)
        validate_prediction_bridge_dtype(values, np)
        self.assertTrue(raw_summary(values, np)['requiresProductionRefusal'])

    def test_valid_photo_domain_promoted_half_values_pass_both_checks(self):
        values = np.asarray([-.25, 0, .125, .5, 1, 1.25], dtype=np.float16).astype(np.float32)
        validate_prediction_bridge_dtype(values, np)
        self.assertFalse(raw_summary(values, np)['requiresProductionRefusal'])


if __name__ == '__main__':
    unittest.main()
