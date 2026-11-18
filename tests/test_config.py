"""配置校验测试：非法参数必须在启动时失败，而不是运行到一半。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from supportpilot.config import Settings


def test_settings_rejects_overlap_not_less_than_size() -> None:
    with pytest.raises(ValidationError):
        Settings(fixed_size=100, fixed_overlap=100)


def test_settings_rejects_min_greater_than_max() -> None:
    with pytest.raises(ValidationError):
        Settings(min_chars=400, max_chars=300)


def test_settings_defaults_are_usable() -> None:
    settings = Settings()

    assert settings.fixed_overlap < settings.fixed_size
    assert settings.min_chars <= settings.max_chars
    assert settings.corpus_dir.is_absolute()
