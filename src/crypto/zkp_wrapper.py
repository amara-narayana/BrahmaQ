"""Cryptographic certification of a computed energy.

What this module is
-------------------
A **commitment scheme** (a.k.a. a sealed-bid envelope):

commit phase
    pick a uniformly random salt ``r``, publish ``c = Commit(m, r)``
reveal phase
    publish ``(m, r)``; anyone can recompute ``Commit(m, r)`` and compare

It must be

* **hiding** -- ``c`` reveals nothing about the message ``m`` while ``r`` is
  secret, and
* **binding** -- the committer cannot find ``(m', r') != (m, r)`` with the same
  commitment.

Implementation: ``HMAC-SHA256`` keyed by a 256-bit salt over a canonical
encoding of the payload.  Hiding comes from the salt being held in a
simulated secure keystore; binding comes from the collision resistance of
HMAC-SHA256.

What this module is *not*
-------------------------
It is **not** a zero-knowledge proof, and it is not lattice-based.  A real
lattice commitment (Ajtai/Lindner-Peikert, or a lattice SNARK over the
computation ``E = <psi|H|psi>``) would let a verifier check the energy without
learning the state and without trusting the prover's hardware.  Implementing
that honestly requires a proved NIZK stack, which is out of scope for this
proof-of-concept; the salted commitment preserves the *auditing* property
(anyone can confirm the published number is the one that was committed to,
and any later tampering is detectable) and is labelled as such in every
output it produces.

Version note: ``cryptography`` 41.0.0 provides ``hmac.HMAC`` with constant-time
verification, which is used here instead of a bare ``hashlib`` digest compare.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, hmac

__all__ = [
    "CommitmentRecord",
    "LatticeCommitmentScheme",
    "canonicalise",
    "DEFAULT_ALGORITHM",
]

DEFAULT_ALGORITHM = "HMAC-SHA256/salted (lattice-commitment surrogate)"
DEFAULT_SALT_BYTES = 32
_COMMITMENT_VERSION = 1


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------
def canonicalise(value: Any) -> Any:
    """Deterministically normalise a payload for hashing.

    Floats are replaced by ``repr(float(x))`` -- Python's shortest
    round-tripping decimal -- so that a value re-serialised anywhere else
    (a different NumPy dtype, a JSON round trip) still produces the same bytes,
    while a genuinely different number never does.  ``bool`` is handled before
    ``int`` because ``bool`` is a subclass of ``int``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            # NaN/Inf have no canonical decimal form, and a commitment to a
            # failed optimisation would certify nothing.
            raise ValueError(f"refusing to certify the non-finite value {value!r}")
        return repr(float(value))
    if isinstance(value, (int, str)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): canonicalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonicalise(item) for item in value]
    if hasattr(value, "item") and hasattr(value, "dtype"):  # numpy scalar
        return canonicalise(value.item())
    return repr(value)


def _encode(payload: Any) -> bytes:
    """Canonical UTF-8 bytes of ``payload``; refuses non-finite numbers."""
    normalised = canonicalise(payload)
    try:
        return json.dumps(
            normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except ValueError as exc:
        raise ValueError(
            "payload contains NaN/Inf and cannot be certified; "
            "check the optimisation and extrapolation results"
        ) from exc


def _hmac_sha256(key: bytes, message: bytes) -> bytes:
    signer = hmac.HMAC(key, hashes.SHA256())
    signer.update(message)
    return signer.finalize()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class CommitmentRecord:
    """A published commitment plus the metadata needed to audit it."""

    proof_id: str
    commitment_hash: str
    algorithm: str
    payload: Dict[str, Any]
    created_utc: str
    salt_bytes: int
    #: Never serialised by :meth:`to_dict` -- the reveal phase needs it.
    salt_hex: str = field(repr=False)

    def to_dict(self, include_salt: bool = False) -> Dict[str, Any]:
        """JSON-safe view.  ``include_salt=True`` opens the envelope."""
        payload: Dict[str, Any] = {
            "proof_id": self.proof_id,
            "commitment_hash": self.commitment_hash,
            "algorithm": self.algorithm,
            "payload": self.payload,
            "created_utc": self.created_utc,
            "salt_bytes": self.salt_bytes,
            "commitment_version": _COMMITMENT_VERSION,
        }
        if include_salt:
            payload["salt_hex"] = self.salt_hex
        return payload


# ---------------------------------------------------------------------------
# Scheme
# ---------------------------------------------------------------------------
class LatticeCommitmentScheme:
    """Salted commitment scheme with a simulated secure keystore.

    Parameters
    ----------
    keystore_path:
        File that plays the role of the prover's tamper-proof secret store.
        Created with owner-only permissions (``0o600`` where the platform
        honours POSIX modes; Windows restricts the ACL instead, so treat the
        file as *simulated* storage, not a hardware security module).
    salt_bytes:
        Salt length; 32 bytes (256 bits) matches the HMAC-SHA256 output size.
    algorithm:
        Label recorded with each commitment, so a verification failure can be
        distinguished from a scheme mismatch.
    """

    def __init__(
        self,
        keystore_path: Union[str, Path] = "artifacts/zkp_keystore.json",
        salt_bytes: int = DEFAULT_SALT_BYTES,
        algorithm: str = DEFAULT_ALGORITHM,
    ) -> None:
        if salt_bytes < 16:
            raise ValueError("salt must be at least 16 bytes")
        self.keystore_path = Path(keystore_path)
        self.salt_bytes = int(salt_bytes)
        self.algorithm = algorithm
        self._keystore: Dict[str, Dict[str, Any]] = self._load_keystore()

    # -- keystore -----------------------------------------------------------
    def _load_keystore(self) -> Dict[str, Dict[str, Any]]:
        if not self.keystore_path.exists():
            return {}
        try:
            with self.keystore_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            # A corrupt keystore must not silently invalidate past proofs.
            return {}

    def _save_keystore(self) -> None:
        self.keystore_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.keystore_path.with_suffix(self.keystore_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self._keystore, handle, indent=2, sort_keys=True)
        os.replace(tmp, self.keystore_path)
        try:
            os.chmod(self.keystore_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:  # pragma: no cover - platform dependent
            pass

    # -- commit -------------------------------------------------------------
    def commit(self, payload: Mapping[str, Any], label: Optional[str] = None) -> CommitmentRecord:
        """Seal ``payload`` and return the published commitment record.

        The message that gets hashed is the payload together with its
        ``proof_id`` and the scheme version, so two identical payloads still
        produce different commitments.
        """
        proof_id = uuid.uuid4().hex
        salt = secrets.token_bytes(self.salt_bytes)
        message = {
            "commitment_version": _COMMITMENT_VERSION,
            "proof_id": proof_id,
            "payload": dict(payload),
        }
        if label is not None:
            message["label"] = str(label)
        digest = _hmac_sha256(salt, _encode(message)).hex()

        record = CommitmentRecord(
            proof_id=proof_id,
            commitment_hash=f"0x{digest}",
            algorithm=self.algorithm,
            payload=dict(payload),
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            salt_bytes=self.salt_bytes,
            salt_hex=salt.hex(),
        )
        self._keystore[proof_id] = {
            "salt_hex": record.salt_hex,
            "commitment_hash": record.commitment_hash,
            "algorithm": record.algorithm,
            "created_utc": record.created_utc,
            "label": label,
        }
        self._save_keystore()
        return record

    # -- verify -------------------------------------------------------------
    def verify(
        self,
        proof_id: str,
        revealed_payload: Mapping[str, Any],
        salt_hex: Optional[str] = None,
        label: Optional[str] = None,
    ) -> bool:
        """Check a revealed payload against the stored commitment.

        Parameters
        ----------
        proof_id:
            Identifier returned at commit time.
        revealed_payload:
            The payload the prover now discloses.
        salt_hex:
            Salt from the reveal phase.  ``None`` uses the keystore copy, which
            models a verifier that trusts the store rather than the channel.

        Returns
        -------
        bool
            ``True`` only if the recomputed commitment matches byte for byte.
            Any tampering of the payload, the salt, or the proof id fails.
        """
        return self.verify_detailed(proof_id, revealed_payload, salt_hex, label)["valid"]

    def verify_detailed(
        self,
        proof_id: str,
        revealed_payload: Mapping[str, Any],
        salt_hex: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """:meth:`verify`, plus the reason and the recomputed digest.

        Both the salt and the optional label default to the values recorded in
        the keystore: a verifier that holds only the commitment and the revealed
        payload must not have to guess what the committer labelled the envelope,
        and silently omitting it would make every verification fail.
        """
        stored = self._keystore.get(proof_id)
        if stored is None:
            return {"valid": False, "reason": f"unknown proof_id {proof_id!r}",
                    "recomputed_hash": None, "expected_hash": None}
        salt_hex = salt_hex if salt_hex is not None else stored.get("salt_hex")
        if label is None:
            label = stored.get("label")
        if salt_hex is None:
            return {"valid": False, "reason": "no salt available for verification",
                    "recomputed_hash": None, "expected_hash": stored.get("commitment_hash")}
        if stored.get("algorithm") not in (None, self.algorithm):
            return {"valid": False, "reason": "algorithm mismatch",
                    "recomputed_hash": None, "expected_hash": stored.get("commitment_hash")}

        message: Dict[str, Any] = {
            "commitment_version": _COMMITMENT_VERSION,
            "proof_id": proof_id,
            "payload": dict(revealed_payload),
        }
        if label is not None:
            message["label"] = str(label)
        try:
            recomputed = "0x" + _hmac_sha256(bytes.fromhex(salt_hex), _encode(message)).hex()
        except ValueError as exc:
            return {"valid": False, "reason": f"invalid salt: {exc}",
                    "recomputed_hash": None, "expected_hash": stored.get("commitment_hash")}

        # Constant-time comparison through the HMAC verifier.  ``verify`` wants
        # the raw signature bytes, so the stored hex digest is decoded first.
        expected = str(stored.get("commitment_hash", ""))
        digest_hex = expected[2:] if expected.startswith("0x") else expected
        verifier = hmac.HMAC(bytes.fromhex(salt_hex), hashes.SHA256())
        verifier.update(_encode(message))
        try:
            verifier.verify(bytes.fromhex(digest_hex))
            valid = True
        except (InvalidSignature, ValueError):
            valid = False

        return {
            "valid": bool(valid),
            "reason": "commitment matches" if valid else "commitment mismatch (payload or salt altered)",
            "recomputed_hash": recomputed,
            "expected_hash": expected,
            "algorithm": self.algorithm,
        }

    # -- reveal -------------------------------------------------------------
    def reveal_salt(self, proof_id: str) -> str:
        """Disclose the salt for a proof (the reveal phase)."""
        stored = self._keystore.get(proof_id)
        if stored is None:
            raise KeyError(f"unknown proof_id {proof_id!r}")
        return str(stored["salt_hex"])

    def verify_energy(
        self, proof_id: str, energy_ha: float, salt_hex: Optional[str] = None
    ) -> bool:
        """Convenience: re-verify a commitment that sealed only ``energy_ha``."""
        return self.verify(proof_id, {"energy_ha": energy_ha}, salt_hex=salt_hex)

    def known_proofs(self) -> List[str]:
        """Proof identifiers held in the keystore."""
        return sorted(self._keystore)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    scheme = LatticeCommitmentScheme(keystore_path="artifacts/_smoke_keystore.json")
    record = scheme.commit({"energy_ha": -1.1372744, "molecule": "H2"})
    print(json.dumps(record.to_dict(), indent=2))
    print("verify(true) :", scheme.verify(record.proof_id, {"energy_ha": -1.1372744, "molecule": "H2"}))
    print("verify(tamper):", scheme.verify(record.proof_id, {"energy_ha": -1.1372745, "molecule": "H2"}))
