from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import json

from .interface.contract import ContractConfig


@dataclass(frozen=True)
class CITTrainerConfig:
    """Configuration for CIT vocabulary induction and compilation.

    This config is intentionally model-agnostic: it defines the *tokenization
    interface* build parameters (budget, candidate extraction, induction proxy).
    The interface contract (serialization/hygiene) is captured separately via
    :class:`ContractConfig` and stored in the artifact.
    """

    vocab_size: int = 8192
    min_freq: int = 10
    len_min: Optional[int] = None
    len_max: int = 24
    boundaries: Optional[list[str]] = None
    preset: str = "default"
    lambda_rd: float = 0.0
    seed: int = 0
    sample_texts: Optional[int] = 200_000

    # Distortion proxy options
    distortion_mode: str = "none"  # 'none' or 'boundary_penalty'
    boundary_penalty: float = 1.0
    include_char_vocab: Optional[bool] = None
    # If True, inject a fixed ASCII fallback alphabet into the vocab so that
    # greedy matching can always fall back to single-character tokens instead
    # of emitting [UNK]. Strongly recommended for structured/security domains.
    include_ascii_fallback: Optional[bool] = None
    symbol_ngram_min_len: int = 2
    symbol_ngram_max_len: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CITTrainerConfig":
        return CITTrainerConfig(**d)


@dataclass(frozen=True)
class CITBuildConfig:
    """Unified build schema persisted into artifacts for reproducibility/audit."""

    schema_version: str = "cit_build.v1"
    trainer: CITTrainerConfig = CITTrainerConfig()
    contract: ContractConfig = ContractConfig()
    # Optional IO metadata (does not affect runtime mapping)
    corpus_format: Optional[str] = None
    text_key: Optional[str] = None
    max_samples: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict will recurse dataclasses properly
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CITBuildConfig":
        trainer = CITTrainerConfig.from_dict(d.get("trainer", {}))
        contract = ContractConfig.from_dict(d.get("contract", {}))
        return CITBuildConfig(
            schema_version=d.get("schema_version", "cit_build.v1"),
            trainer=trainer,
            contract=contract,
            corpus_format=d.get("corpus_format"),
            text_key=d.get("text_key"),
            max_samples=d.get("max_samples"),
        )

    @staticmethod
    def from_json(s: str) -> "CITBuildConfig":
        return CITBuildConfig.from_dict(json.loads(s))
