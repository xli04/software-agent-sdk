"""Helpers for populating ``ToolShieldLLMSecurityAnalyzer.safety_experiences``.

These helpers integrate with the ``toolshield`` PyPI package (install via the
``[toolshield]`` optional extra). They expose three usage patterns:

1. :func:`default_safety_experiences` -- seed with terminal + filesystem
   experiences we ship by default.
2. :func:`load_safety_experiences` -- load an explicit list of tool
   experiences.
3. :func:`auto_detect_safety_experiences` -- probe localhost for active MCP
   servers, load experiences for the tools that are actually running.

All three return a rendered string ready to plug into
``ToolShieldLLMSecurityAnalyzer(safety_experiences=...)``. Users who want to
inject their own hand-authored experiences can skip these helpers and pass
an arbitrary string directly.

Example:
    >>> from openhands.sdk.security import ToolShieldLLMSecurityAnalyzer
    >>> from openhands.sdk.security.toolshield_helpers import (
    ...     default_safety_experiences,
    ...     auto_detect_safety_experiences,
    ... )
    >>>
    >>> # Default seed
    >>> analyzer = ToolShieldLLMSecurityAnalyzer(
    ...     llm=guardrail_llm,
    ...     safety_experiences=default_safety_experiences(),
    ... )
    >>>
    >>> # Auto-detect whatever MCP servers are running locally
    >>> analyzer = ToolShieldLLMSecurityAnalyzer(
    ...     llm=guardrail_llm,
    ...     safety_experiences=auto_detect_safety_experiences(),
    ... )
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

from openhands.sdk.logger import get_logger


if TYPE_CHECKING:
    # Only for type hints; keep the real import lazy so the SDK doesn't
    # require toolshield to be installed.
    from toolshield import ExperienceStore  # noqa: F401


logger = get_logger(__name__)


# Tools seeded by default. These are the ones we have bundled experiences for
# and that cover the tool surface evaluated in the linked issue.
DEFAULT_TOOL_NAMES: list[str] = ["terminal-mcp", "filesystem-mcp"]


# Known MCP server ports -- the defaults used by our evaluation harness and
# matching the convention in mcpmark/MCP tooling. Callers with non-standard
# ports can pass their own mapping to the auto-detect helper.
MCP_PORT_DEFAULTS: dict[str, int] = {
    "filesystem-mcp": 9090,
    "postgres-mcp": 9091,
    "playwright-mcp": 9092,
    "notion-mcp": 9097,
}


# Tools that don't have a port to probe (terminal is local exec). We include
# them unconditionally in auto-detect results.
ALWAYS_ACTIVE_TOOLS: list[str] = ["terminal-mcp"]


def _require_toolshield():
    """Import the toolshield package or raise a helpful ImportError."""
    try:
        import toolshield  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "toolshield is not installed. Install via "
            "`pip install openhands-sdk[toolshield]` to use these helpers, "
            "or pass a custom string to "
            "ToolShieldLLMSecurityAnalyzer(safety_experiences=...)."
        ) from e
    return toolshield


def load_safety_experiences(
    tool_names: list[str],
    model: str = "claude-sonnet-4.5",
) -> str:
    """Load experiences for an explicit list of tool names.

    Args:
        tool_names: Tool experience identifiers (e.g. ``"terminal-mcp"``).
            Must match a file bundled in the ``toolshield`` package for the
            given ``model`` subdirectory.
        model: Which pre-generated experience set to use. Defaults to
            ``"claude-sonnet-4.5"``.

    Returns:
        A rendered string ready for ``safety_experiences=``.
    """
    ts = _require_toolshield()
    experiences = ts.load_experiences(tool_names, model=model)
    return experiences.format_for_prompt()


def default_safety_experiences(model: str = "claude-sonnet-4.5") -> str:
    """Default seed: terminal + filesystem experiences.

    This is the starting point that covers the tool surface evaluated in the
    linked issue. Callers with different tool surfaces should use
    :func:`load_safety_experiences` or :func:`auto_detect_safety_experiences`
    instead.
    """
    return load_safety_experiences(DEFAULT_TOOL_NAMES, model=model)


def _probe_port(host: str, port: int, timeout: float = 0.5) -> bool:
    """Quick TCP probe: return True if something is listening."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def detect_active_mcp_tools(
    host: str = "localhost",
    port_map: dict[str, int] | None = None,
    timeout: float = 0.5,
) -> list[str]:
    """Scan ``host`` and return tool names whose MCP servers are listening.

    Tools in :data:`ALWAYS_ACTIVE_TOOLS` (terminal) are returned
    unconditionally since they're local exec rather than network services.

    Args:
        host: Host to probe. Defaults to ``"localhost"``.
        port_map: Mapping from tool name to port. Defaults to
            :data:`MCP_PORT_DEFAULTS`. Pass a custom dict if you run MCP
            servers on non-standard ports.
        timeout: Per-port TCP timeout in seconds.

    Returns:
        List of tool names whose servers are responsive, in deterministic
        order (always-active tools first, then port-probed tools).
    """
    port_map = port_map if port_map is not None else MCP_PORT_DEFAULTS
    active = list(ALWAYS_ACTIVE_TOOLS)
    for tool_name, port in port_map.items():
        if _probe_port(host, port, timeout=timeout):
            active.append(tool_name)
            logger.debug(
                f"MCP server active at {host}:{port} -> {tool_name}"
            )
        else:
            logger.debug(
                f"No MCP server at {host}:{port} (skipping {tool_name})"
            )
    return active


def auto_detect_safety_experiences(
    host: str = "localhost",
    port_map: dict[str, int] | None = None,
    timeout: float = 0.5,
    model: str = "claude-sonnet-4.5",
    fallback_to_default: bool = True,
) -> str:
    """Scan ``host`` for active MCP servers and load their experiences.

    "Detection" requires at least one *networked* MCP server to respond --
    the unconditionally-included always-active tools (e.g. terminal) don't
    count as detection. When no networked tool is detected, falls back to
    :func:`default_safety_experiences` (terminal + filesystem), unless
    ``fallback_to_default=False``, in which case returns an empty string so
    the caller's no-op path doesn't quietly require ``toolshield``.

    Args:
        host: Host to probe. Defaults to ``"localhost"``.
        port_map: Tool-to-port mapping; see :func:`detect_active_mcp_tools`.
        timeout: Per-port TCP timeout in seconds.
        model: Experience-set subdirectory. Defaults to
            ``"claude-sonnet-4.5"``.
        fallback_to_default: If no MCP servers are detected, return the
            default seed (terminal + filesystem) instead of an empty string.

    Returns:
        A rendered string ready for ``safety_experiences=``.
    """
    active = detect_active_mcp_tools(
        host=host, port_map=port_map, timeout=timeout
    )
    # "Detection" means at least one *networked* MCP server responded. The
    # always-active tools (e.g., terminal) don't count as detection signal
    # because they're included unconditionally.
    networked_detected = [t for t in active if t not in ALWAYS_ACTIVE_TOOLS]

    if networked_detected:
        logger.info(f"Auto-detected active MCP tools: {active}")
        return load_safety_experiences(active, model=model)

    if fallback_to_default:
        logger.info(
            "No networked MCP tools detected; falling back to default seed "
            f"({DEFAULT_TOOL_NAMES})"
        )
        return default_safety_experiences(model=model)

    # Caller explicitly opted out of the default seed. Return empty so they
    # don't silently pick up terminal-mcp experiences (and don't require
    # ``toolshield`` to be installed for the no-op path).
    logger.warning(
        "No networked MCP tools detected and fallback_to_default=False; "
        "returning empty safety_experiences"
    )
    return ""
