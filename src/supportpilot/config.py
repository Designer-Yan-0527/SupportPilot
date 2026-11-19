"""集中配置：所有可调参数从这里读取，代码里不再出现魔法数字。

优先级：环境变量 > .env 文件 > 这里的默认值。
环境变量统一加前缀 SUPPORTPILOT_，例如 SUPPORTPILOT_FIXED_SIZE=250。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 由文件位置反推项目根目录，避免依赖启动命令所在目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FIXED_SIZE = 300
DEFAULT_FIXED_OVERLAP = 50
DEFAULT_MAX_CHARS = 300
DEFAULT_MIN_CHARS = 120


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPPORTPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    questions_file: Path = PROJECT_ROOT / "data" / "eval" / "questions.json"
    results_dir: Path = PROJECT_ROOT / "artifacts" / "eval"
    qdrant_dir: Path = PROJECT_ROOT / "artifacts" / "qdrant"

    fixed_size: int = Field(DEFAULT_FIXED_SIZE, ge=1)
    fixed_overlap: int = Field(DEFAULT_FIXED_OVERLAP, ge=0)
    max_chars: int = Field(DEFAULT_MAX_CHARS, ge=1)
    min_chars: int = Field(DEFAULT_MIN_CHARS, ge=0)

    @model_validator(mode="after")
    def _check_ranges(self) -> Settings:
        # 这两个约束决定了切分循环一定能前进，启动时就校验，别等跑一半才卡死
        if self.fixed_overlap >= self.fixed_size:
            raise ValueError("fixed_overlap 必须小于 fixed_size，否则切分步长非正")
        if self.min_chars > self.max_chars:
            raise ValueError("min_chars 不能大于 max_chars")
        return self


@lru_cache
def get_settings() -> Settings:
    """进程内只读一次配置；测试里可用 get_settings.cache_clear() 重置。"""
    return Settings()
