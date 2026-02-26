"""Placeholder unit tests to verify the project scaffold is functional.

These tests exist to ensure:
- The package is importable after ``pip install -e .``
- ``__version__`` is set correctly
- The plugin registry operates correctly at a basic level
- The CI pipeline is green from the first commit

Replace or extend these tests with real domain logic tests as the
project grows.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

import agent_sim_bridge
from agent_sim_bridge.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)


# ---------------------------------------------------------------------------
# Package-level tests
# ---------------------------------------------------------------------------


class TestPackageMetadata:
    def test_version_is_string(self) -> None:
        assert isinstance(agent_sim_bridge.__version__, str)

    def test_version_has_three_parts(self) -> None:
        parts = agent_sim_bridge.__version__.split(".")
        assert len(parts) == 3, (
            f"Expected semver X.Y.Z, got {agent_sim_bridge.__version__!r}"
        )

    def test_version_matches_expected(self, expected_version: str) -> None:
        assert agent_sim_bridge.__version__ == expected_version

    def test_version_exported_in_all(self) -> None:
        assert "__version__" in agent_sim_bridge.__all__


# ---------------------------------------------------------------------------
# Plugin registry tests
# ---------------------------------------------------------------------------


class BaseWidget(ABC):
    @abstractmethod
    def render(self) -> str: ...


@pytest.fixture()
def widget_registry() -> PluginRegistry[BaseWidget]:
    """Fresh registry for each test to prevent state leakage."""
    return PluginRegistry(BaseWidget, "widgets")


class TestPluginRegistryRegistration:
    def test_register_and_retrieve(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        @widget_registry.register("basic")
        class BasicWidget(BaseWidget):
            def render(self) -> str:
                return "<basic/>"

        cls = widget_registry.get("basic")
        assert cls is BasicWidget

    def test_registered_plugin_is_instantiable(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        @widget_registry.register("instant")
        class InstantWidget(BaseWidget):
            def render(self) -> str:
                return "ok"

        instance = widget_registry.get("instant")()
        assert instance.render() == "ok"

    def test_register_class_directly(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        class DirectWidget(BaseWidget):
            def render(self) -> str:
                return "direct"

        widget_registry.register_class("direct", DirectWidget)
        assert widget_registry.get("direct") is DirectWidget

    def test_duplicate_registration_raises(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        @widget_registry.register("dup")
        class FirstWidget(BaseWidget):
            def render(self) -> str:
                return "first"

        with pytest.raises(PluginAlreadyRegisteredError) as exc_info:

            @widget_registry.register("dup")
            class SecondWidget(BaseWidget):
                def render(self) -> str:
                    return "second"

        assert exc_info.value.plugin_name == "dup"
        assert exc_info.value.registry_name == "widgets"

    def test_non_subclass_raises_type_error(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        class NotAWidget:
            pass

        with pytest.raises(TypeError, match="must be a subclass"):
            widget_registry.register_class("bad", NotAWidget)  # type: ignore[arg-type]

    def test_deregister_removes_plugin(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        @widget_registry.register("transient")
        class TransientWidget(BaseWidget):
            def render(self) -> str:
                return "transient"

        widget_registry.deregister("transient")
        assert "transient" not in widget_registry

    def test_deregister_missing_raises(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        with pytest.raises(PluginNotFoundError):
            widget_registry.deregister("ghost")


class TestPluginRegistryLookup:
    def test_get_missing_raises(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        with pytest.raises(PluginNotFoundError) as exc_info:
            widget_registry.get("missing")
        assert exc_info.value.plugin_name == "missing"

    def test_list_plugins_empty(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        assert widget_registry.list_plugins() == []

    def test_list_plugins_sorted(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        class ZebraWidget(BaseWidget):
            def render(self) -> str:
                return "zebra"

        class AlphaWidget(BaseWidget):
            def render(self) -> str:
                return "alpha"

        class MangoWidget(BaseWidget):
            def render(self) -> str:
                return "mango"

        widget_registry.register_class("zebra", ZebraWidget)
        widget_registry.register_class("alpha", AlphaWidget)
        widget_registry.register_class("mango", MangoWidget)

        assert widget_registry.list_plugins() == ["alpha", "mango", "zebra"]

    def test_contains_operator(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        @widget_registry.register("present")
        class PresentWidget(BaseWidget):
            def render(self) -> str:
                return "here"

        assert "present" in widget_registry
        assert "absent" not in widget_registry

    def test_len(self, widget_registry: PluginRegistry[BaseWidget]) -> None:
        assert len(widget_registry) == 0

        @widget_registry.register("one")
        class OneWidget(BaseWidget):
            def render(self) -> str:
                return "1"

        assert len(widget_registry) == 1

    def test_repr_contains_registry_name(
        self, widget_registry: PluginRegistry[BaseWidget]
    ) -> None:
        assert "widgets" in repr(widget_registry)
