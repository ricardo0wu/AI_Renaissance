"""AgentScope message bridge for AI Renaissance runtime contracts.

The project-level business contract remains `Signal`. AgentScope `Msg`
is only the runtime transport contract at framework boundaries.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.signal import Signal


SIGNAL_METADATA_KEY = "signal"
ARBITRATION_RESULT_METADATA_KEY = "arbitration_result"
MESSAGE_TYPE_METADATA_KEY = "message_type"
TASK_METADATA_KEY = "task"
STOCK_CODE_METADATA_KEY = "stock_code"
DEFAULT_STOCK_ANALYSIS_TASK = "analyze_stock"
STOCK_ANALYSIS_TASK_MESSAGE_TYPE = "stock_analysis_task"
EXPERT_SIGNAL_MESSAGE_TYPE = "expert_signal"
ARBITRATION_RESULT_MESSAGE_TYPE = "arbitration_result"


class AgentScopeMessageError(ValueError):
    """Raised when an AgentScope message cannot satisfy project contracts."""


def _create_msg(
    *,
    name: str,
    role: str,
    content: str,
    metadata: Dict[str, Any],
    msg_class: Optional[Any] = None,
):
    if msg_class is not None:
        return msg_class(
            name=name,
            role=role,
            content=content,
            metadata=metadata,
        )

    try:
        from agentscope.message import AssistantMsg, SystemMsg, UserMsg
    except Exception as exc:  # pragma: no cover - depends on AgentScope runtime import
        raise AgentScopeMessageError(f"AgentScope message helpers are unavailable: {exc}") from exc

    if role == "user":
        return UserMsg(name=name, content=content, metadata=metadata)
    if role == "assistant":
        return AssistantMsg(name=name, content=content, metadata=metadata)
    if role == "system":
        return SystemMsg(name=name, content=content, metadata=metadata)
    raise AgentScopeMessageError(f"Unsupported AgentScope message role: {role}")


def _metadata(msg: Any) -> Dict[str, Any]:
    metadata = getattr(msg, "metadata", None)
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise AgentScopeMessageError("AgentScope Msg metadata must be a dict")
    return metadata


def stock_task_to_msg(
    stock_code: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    name: str = "OrchestratorAgent",
    msg_class: Optional[Any] = None,
):
    """Create an AgentScope Msg for a stock analysis task."""
    code = (stock_code or "").strip()
    if not code:
        raise AgentScopeMessageError("stock_code is required")

    metadata = {
        MESSAGE_TYPE_METADATA_KEY: STOCK_ANALYSIS_TASK_MESSAGE_TYPE,
        TASK_METADATA_KEY: DEFAULT_STOCK_ANALYSIS_TASK,
        STOCK_CODE_METADATA_KEY: code,
        "context": context or {},
    }
    return _create_msg(
        name=name,
        role="user",
        content=f"Analyze stock {code}",
        metadata=metadata,
        msg_class=msg_class,
    )


def signal_to_msg(
    signal: Signal,
    *,
    name: Optional[str] = None,
    msg_class: Optional[Any] = None,
):
    """Wrap a project Signal in an AgentScope Msg."""
    if not isinstance(signal, Signal):
        raise AgentScopeMessageError(f"Expected Signal, got {type(signal).__name__}")

    return _create_msg(
        name=name or signal.source or signal.signal_type or "ExpertAgent",
        role="assistant",
        content=signal.reasoning,
        metadata={
            MESSAGE_TYPE_METADATA_KEY: EXPERT_SIGNAL_MESSAGE_TYPE,
            SIGNAL_METADATA_KEY: signal.to_dict(),
            "signal_type": signal.signal_type,
            "source": signal.source,
        },
        msg_class=msg_class,
    )


def msg_to_signal(msg: Any) -> Signal:
    """Extract a project Signal from an AgentScope Msg."""
    data = _metadata(msg).get(SIGNAL_METADATA_KEY)
    if data is None:
        raise AgentScopeMessageError("AgentScope Msg metadata does not contain a signal")
    if not isinstance(data, dict):
        raise AgentScopeMessageError("AgentScope Msg signal metadata must be a dict")
    try:
        return Signal.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentScopeMessageError(f"AgentScope Msg signal metadata is invalid: {exc}") from exc


def arbitration_result_to_msg(
    result: Any,
    *,
    name: str = "OrchestratorAgent",
    msg_class: Optional[Any] = None,
):
    """Wrap an Orchestrator ArbitrationResult in an AgentScope Msg."""
    required_fields = [
        "decision",
        "direction",
        "confidence",
        "position_ratio",
        "reasoning",
        "signals_summary",
        "risks",
        "reasoning_chain",
    ]
    missing = [field for field in required_fields if not hasattr(result, field)]
    if missing:
        raise AgentScopeMessageError(
            f"Expected ArbitrationResult-like object, missing fields: {missing}",
        )

    data = {field: getattr(result, field) for field in required_fields}
    return _create_msg(
        name=name,
        role="assistant",
        content=result.reasoning,
        metadata={
            MESSAGE_TYPE_METADATA_KEY: ARBITRATION_RESULT_MESSAGE_TYPE,
            ARBITRATION_RESULT_METADATA_KEY: data,
            "scope_trace": result.scope_trace,
            "decision": result.decision,
            "direction": result.direction,
        },
        msg_class=msg_class,
    )


def extract_stock_code(msg: Any) -> str:
    """Read the stock code from an AgentScope task Msg."""
    metadata = _metadata(msg)
    stock_code = metadata.get(STOCK_CODE_METADATA_KEY)

    if not stock_code and isinstance(getattr(msg, "content", None), dict):
        stock_code = msg.content.get(STOCK_CODE_METADATA_KEY)

    if not isinstance(stock_code, str) or not stock_code.strip():
        raise AgentScopeMessageError("AgentScope Msg does not contain stock_code")
    return stock_code.strip()
