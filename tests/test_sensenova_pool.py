import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sensenova_pool import (  # noqa: E402
    DEFAULT_POOL_BASE_URL,
    DEFAULT_POOL_LOCAL_TOKEN,
    load_sensenova_pool_config,
)


class SenseNovaPoolConfigTests(unittest.TestCase):
    def test_loads_pool_provider_from_opencode_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "opencode.jsonc"
            path.write_text(
                "// retained comment\n"
                + json.dumps(
                    {
                        "provider": {
                            "sensenova-pool": {
                                "options": {
                                    "baseURL": "http://127.0.0.1:18787/v1",
                                    "apiKey": "local-token",
                                },
                                "models": {"sensenova-6.8-flash-lite": {}},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_sensenova_pool_config({}, path)

        self.assertIsNotNone(config)
        self.assertEqual("http://127.0.0.1:18787/v1", config.base_url)
        self.assertEqual("local-token", config.api_key)
        self.assertEqual("http://127.0.0.1:18787/health", config.health_url)
        self.assertIn("sensenova-6.8-flash-lite", config.models)

    def test_environment_can_disable_configured_pool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "opencode.jsonc"
            path.write_text(
                json.dumps(
                    {
                        "provider": {
                            "sensenova-pool": {
                                "options": {
                                    "baseURL": "http://127.0.0.1:18787/v1",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_sensenova_pool_config(
                {"SENSENOVA_POOL_ENABLED": "0"},
                path,
            )

        self.assertIsNone(config)

    def test_explicit_enable_uses_safe_local_defaults_without_opencode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.jsonc"
            config = load_sensenova_pool_config(
                {"SENSENOVA_POOL_ENABLED": "1"},
                missing,
            )

        self.assertIsNotNone(config)
        self.assertEqual(DEFAULT_POOL_BASE_URL, config.base_url)
        self.assertEqual(DEFAULT_POOL_LOCAL_TOKEN, config.api_key)


if __name__ == "__main__":
    unittest.main()
