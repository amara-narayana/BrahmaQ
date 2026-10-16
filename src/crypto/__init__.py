"""Cryptographic layer: salted commitment schemes for energy certification."""

from .zkp_wrapper import (
    CommitmentRecord,
    LatticeCommitmentScheme,
    canonicalise,
)

__all__ = ["CommitmentRecord", "LatticeCommitmentScheme", "canonicalise"]
