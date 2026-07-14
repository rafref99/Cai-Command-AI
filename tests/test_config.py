from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cai.config import (
    DEFAULT_PROFILE,
    AppConfig,
    api_key_setting_is_encoded,
    apply_env_overrides,
    decode_api_key_setting,
    list_profiles,
    load_config,
    profile_config_path,
    read_active_profile,
    save_config,
    selected_profile,
    validate_profile_name,
    write_active_profile,
)


class ProfileConfigTests(unittest.TestCase):
    def test_context_budget_environment_override_is_coerced(self) -> None:
        with patch.dict("os.environ", {"CAI_MAX_CONTEXT_CHARS": "32000"}):
            config = apply_env_overrides(AppConfig())

        self.assertEqual(config.max_context_chars, 32_000)

    def test_invalid_boolean_environment_override_is_not_silently_false(self) -> None:
        with patch.dict("os.environ", {"CAI_DRY_RUN": "tru"}, clear=True):
            config = apply_env_overrides(AppConfig(dry_run=True))

        self.assertEqual(config.dry_run, "tru")

    def test_shell_command_environment_override_clears_saved_argv_command(self) -> None:
        saved = AppConfig(
            command_provider="old-shell",
            command_provider_argv="python stale.py",
        )

        with patch.dict(
            "os.environ",
            {"CAI_COMMAND_PROVIDER": "./new-wrapper"},
            clear=True,
        ):
            config = apply_env_overrides(saved)

        self.assertEqual(config.command_provider, "./new-wrapper")
        self.assertEqual(config.command_provider_argv, "")

    def test_default_and_named_profile_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            profiles_dir = root / "profiles"

            self.assertEqual(
                profile_config_path(
                    DEFAULT_PROFILE,
                    config_path=config_path,
                    profiles_dir=profiles_dir,
                ),
                config_path,
            )
            self.assertEqual(
                profile_config_path(
                    "local-gemma",
                    config_path=config_path,
                    profiles_dir=profiles_dir,
                ),
                profiles_dir / "local-gemma.json",
            )

    def test_profile_names_reject_path_traversal(self) -> None:
        for name in {"../secret", "nested/profile", ".hidden", ""}:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_profile_name(name)

    def test_active_profile_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active_profile"

            self.assertEqual(read_active_profile(path), DEFAULT_PROFILE)
            write_active_profile("hosted-api", path)

            self.assertEqual(read_active_profile(path), "hosted-api")
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_list_profiles_includes_default_and_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles_dir = Path(tmp)
            (profiles_dir / "local.json").write_text("{}", encoding="utf-8")
            (profiles_dir / "hosted.json").write_text("{}", encoding="utf-8")
            (profiles_dir / ".invalid.json").write_text("{}", encoding="utf-8")
            (profiles_dir / "notes.txt").write_text("ignored", encoding="utf-8")

            self.assertEqual(
                list_profiles(profiles_dir),
                [DEFAULT_PROFILE, "hosted", "local"],
            )

    def test_explicit_profile_takes_precedence_over_environment(self) -> None:
        with patch.dict("os.environ", {"CAI_PROFILE": "from-env"}):
            self.assertEqual(selected_profile("explicit"), "explicit")
            self.assertEqual(selected_profile(), "from-env")


class ApiKeyStorageTests(unittest.TestCase):
    def test_uppercase_sk_prefixed_environment_name_is_not_encoded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"

            save_config(AppConfig(api_key_env="SK_TOKEN"), path)

            self.assertEqual(load_config(path).api_key_env, "SK_TOKEN")

    def test_lowercase_sk_prefixed_environment_name_is_not_encoded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"

            save_config(AppConfig(api_key_env="sk_token"), path)

            self.assertEqual(load_config(path).api_key_env, "sk_token")

    def test_save_config_base64_encodes_raw_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"

            save_config(AppConfig(api_key_env="sk-local:secret"), path)
            saved = load_config(path)

            self.assertTrue(api_key_setting_is_encoded(saved.api_key_env))
            self.assertNotIn("sk-local:secret", path.read_text(encoding="utf-8"))
            self.assertEqual(decode_api_key_setting(saved.api_key_env), "sk-local:secret")
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_internal_config_directory_uses_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".cai"
            path = root / "config.json"

            with (
                patch("cai.config.CONFIG_DIR", root),
                patch("cai.config.CONFIG_PATH", path),
            ):
                save_config(AppConfig(), path)

            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_raw_prefix_encodes_identifier_shaped_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"

            save_config(AppConfig(api_key_env="raw:gsk_exampletoken"), path)
            saved = load_config(path)

            self.assertTrue(api_key_setting_is_encoded(saved.api_key_env))
            self.assertEqual(decode_api_key_setting(saved.api_key_env), "gsk_exampletoken")

    def test_save_config_keeps_environment_variable_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"

            save_config(AppConfig(api_key_env="OPENAI_API_KEY"), path)

            self.assertEqual(load_config(path).api_key_env, "OPENAI_API_KEY")

    def test_save_config_does_not_double_encode_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(AppConfig(api_key_env="sk-local:secret"), path)
            encoded = load_config(path).api_key_env

            save_config(load_config(path), path)

            self.assertEqual(load_config(path).api_key_env, encoded)

    def test_invalid_base64_api_key_decodes_to_empty(self) -> None:
        self.assertEqual(decode_api_key_setting("base64:not-valid"), "")


if __name__ == "__main__":
    unittest.main()
