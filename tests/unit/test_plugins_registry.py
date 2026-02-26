"""Unit tests for the plugin registry module.

Covers:
- PluginNotFoundError / PluginAlreadyRegisteredError attributes
- PluginRegistry.register (decorator usage, duplicate raises, wrong type)
- PluginRegistry.register_class (direct registration, duplicate, wrong type)
- PluginRegistry.deregister (existing, not-found)
- PluginRegistry.get (found, not-found)
- PluginRegistry.list_plugins (sorted)
- PluginRegistry.__contains__ / __len__ / __repr__
- PluginRegistry.load_entrypoints (already registered idempotent, bad load,
  bad class skipped)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from unittest.mock import MagicMock, patch

import pytest

from agent_sim_bridge.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)


# ---------------------------------------------------------------------------
# Minimal ABC + concrete implementations for tests
# ---------------------------------------------------------------------------


class BasePlugin(ABC):
    @abstractmethod
    def run(self) -> str: ...


class AlphaPlugin(BasePlugin):
    def run(self) -> str:
        return "alpha"


class BetaPlugin(BasePlugin):
    def run(self) -> str:
        return "beta"


class GammaPlugin(BasePlugin):
    def run(self) -> str:
        return "gamma"


class NotAPlugin:
    """Does NOT subclass BasePlugin."""

    pass


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class TestPluginNotFoundError:
    def test_attributes(self) -> None:
        err = PluginNotFoundError("my-plugin", "test-registry")
        assert err.plugin_name == "my-plugin"
        assert err.registry_name == "test-registry"
        assert "my-plugin" in str(err)


class TestPluginAlreadyRegisteredError:
    def test_attributes(self) -> None:
        err = PluginAlreadyRegisteredError("dup", "reg")
        assert err.plugin_name == "dup"
        assert err.registry_name == "reg"
        assert "dup" in str(err)


# ---------------------------------------------------------------------------
# PluginRegistry
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    def setup_method(self) -> None:
        self.registry: PluginRegistry[BasePlugin] = PluginRegistry(BasePlugin, "test-reg")

    # --- register decorator ---

    def test_register_decorator_adds_plugin(self) -> None:
        @self.registry.register("alpha")
        class _Local(BasePlugin):
            def run(self) -> str:
                return "local"

        assert "alpha" in self.registry

    def test_register_decorator_returns_class_unchanged(self) -> None:
        @self.registry.register("ret-test")
        class _Local(BasePlugin):
            def run(self) -> str:
                return "x"

        assert issubclass(_Local, BasePlugin)

    def test_register_duplicate_raises(self) -> None:
        self.registry.register("dup")(AlphaPlugin)
        with pytest.raises(PluginAlreadyRegisteredError):
            self.registry.register("dup")(BetaPlugin)

    def test_register_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="subclass"):
            self.registry.register("bad")(NotAPlugin)  # type: ignore[arg-type]

    # --- register_class ---

    def test_register_class_adds_plugin(self) -> None:
        self.registry.register_class("alpha", AlphaPlugin)
        assert self.registry.get("alpha") is AlphaPlugin

    def test_register_class_duplicate_raises(self) -> None:
        self.registry.register_class("alpha", AlphaPlugin)
        with pytest.raises(PluginAlreadyRegisteredError):
            self.registry.register_class("alpha", BetaPlugin)

    def test_register_class_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError):
            self.registry.register_class("bad", NotAPlugin)  # type: ignore[arg-type]

    # --- deregister ---

    def test_deregister_removes_plugin(self) -> None:
        self.registry.register_class("alpha", AlphaPlugin)
        self.registry.deregister("alpha")
        assert "alpha" not in self.registry

    def test_deregister_not_found_raises(self) -> None:
        with pytest.raises(PluginNotFoundError):
            self.registry.deregister("ghost")

    # --- get ---

    def test_get_returns_class(self) -> None:
        self.registry.register_class("alpha", AlphaPlugin)
        retrieved = self.registry.get("alpha")
        assert retrieved is AlphaPlugin

    def test_get_not_found_raises(self) -> None:
        with pytest.raises(PluginNotFoundError):
            self.registry.get("phantom")

    # --- list_plugins ---

    def test_list_plugins_sorted(self) -> None:
        self.registry.register_class("zebra", AlphaPlugin)
        self.registry.register_class("apple", BetaPlugin)
        assert self.registry.list_plugins() == ["apple", "zebra"]

    def test_list_plugins_empty(self) -> None:
        assert self.registry.list_plugins() == []

    # --- membership / len / repr ---

    def test_contains_registered(self) -> None:
        self.registry.register_class("alpha", AlphaPlugin)
        assert "alpha" in self.registry

    def test_contains_not_registered(self) -> None:
        assert "missing" not in self.registry

    def test_len_empty(self) -> None:
        assert len(self.registry) == 0

    def test_len_after_register(self) -> None:
        self.registry.register_class("alpha", AlphaPlugin)
        assert len(self.registry) == 1

    def test_repr(self) -> None:
        self.registry.register_class("alpha", AlphaPlugin)
        result = repr(self.registry)
        assert "test-reg" in result
        assert "alpha" in result

    # --- load_entrypoints ---

    def test_load_entrypoints_registers_plugin(self) -> None:
        ep = MagicMock()
        ep.name = "ep-alpha"
        ep.load.return_value = AlphaPlugin

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            self.registry.load_entrypoints("some.group")

        assert "ep-alpha" in self.registry

    def test_load_entrypoints_already_registered_skipped(self) -> None:
        self.registry.register_class("ep-alpha", AlphaPlugin)

        ep = MagicMock()
        ep.name = "ep-alpha"
        ep.load.return_value = BetaPlugin

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            self.registry.load_entrypoints("some.group")

        # Should still be AlphaPlugin (not replaced).
        assert self.registry.get("ep-alpha") is AlphaPlugin

    def test_load_entrypoints_bad_load_skipped(self) -> None:
        ep = MagicMock()
        ep.name = "bad-ep"
        ep.load.side_effect = ImportError("missing dep")

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            self.registry.load_entrypoints("some.group")

        assert "bad-ep" not in self.registry

    def test_load_entrypoints_wrong_type_skipped(self) -> None:
        ep = MagicMock()
        ep.name = "wrong-ep"
        ep.load.return_value = NotAPlugin

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            self.registry.load_entrypoints("some.group")

        assert "wrong-ep" not in self.registry

    def test_load_entrypoints_multiple(self) -> None:
        eps = []
        for name, cls in [("a", AlphaPlugin), ("b", BetaPlugin), ("c", GammaPlugin)]:
            ep = MagicMock()
            ep.name = name
            ep.load.return_value = cls
            eps.append(ep)

        with patch("importlib.metadata.entry_points", return_value=eps):
            self.registry.load_entrypoints("some.group")

        assert len(self.registry) == 3
