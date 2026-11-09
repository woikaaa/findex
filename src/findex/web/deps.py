"""Залежності FastAPI (Depends): через них маршрути отримують стан і налаштування.

У тестах я підмінюю їх через app.dependency_overrides, тому справжній файл індексу
для тестів не потрібен."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from findex.web.settings import Settings
from findex.web.state import AppState


def get_settings(request: Request) -> Settings:
    """Налаштування, з якими створено застосунок (лежать у app.state)."""
    settings: Settings = request.app.state.settings
    return settings


def get_state_or_none(request: Request) -> AppState | None:
    """Стан сервісу або None, якщо індекс ще не завантажений (або не вдалося)."""
    state: AppState | None = getattr(request.app.state, "findex", None)
    return state


def get_state(state: Annotated[AppState | None, Depends(get_state_or_none)]) -> AppState:
    """Стан сервісу. Якщо індекс ще не завантажений, віддаю 503 (спробуйте пізніше)."""
    if state is None:
        raise HTTPException(status_code=503, detail="Індекс ще не завантажений")
    return state
