"""실주문 시도 기록을 세션에서 누적 관리하는 작은 안전 도우미."""

from collections.abc import MutableMapping

_ATTEMPT_KEY = "attempted_orders"
_MAX_ATTEMPTS = 100


def was_order_attempted(state: MutableMapping, signature: str) -> bool:
    return signature in state.get(_ATTEMPT_KEY, ())


def record_order_attempt(state: MutableMapping, signature: str) -> None:
    attempts = list(state.get(_ATTEMPT_KEY, ()))
    if signature not in attempts:
        attempts.append(signature)
    state[_ATTEMPT_KEY] = attempts[-_MAX_ATTEMPTS:]
