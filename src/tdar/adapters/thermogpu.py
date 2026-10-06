"""Persistent subprocess adapter for the ThermoGPU TDAR oracle."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ThermoGPUDomain:
    """Affine map between TDAR's unit square and a physical T/P rectangle."""

    temperature_K: tuple[float, float] = (250.0, 350.0)
    pressure_Pa: tuple[float, float] = (1.0e6, 10.0e6)

    def __post_init__(self):
        if not self.temperature_K[1] > self.temperature_K[0]:
            raise ValueError("temperature_K bounds must be increasing")
        if not self.pressure_Pa[1] > self.pressure_Pa[0]:
            raise ValueError("pressure_Pa bounds must be increasing")

    @property
    def temperature_span(self) -> float:
        return self.temperature_K[1] - self.temperature_K[0]

    @property
    def pressure_span(self) -> float:
        return self.pressure_Pa[1] - self.pressure_Pa[0]

    def physical(self, uv) -> tuple[float, float]:
        uv = np.asarray(uv, dtype=float)
        if uv.shape != (2,):
            raise ValueError("normalized point must have shape (2,)")
        if not np.all(np.isfinite(uv)):
            raise ValueError("normalized point must be finite")
        t = self.temperature_K[0] + uv[0] * self.temperature_span
        p = self.pressure_Pa[0] + uv[1] * self.pressure_span
        return float(t), float(p)

    def normalized_gradient(self, dZ_dT: float, dZ_dP: float) -> np.ndarray:
        return np.array([
            dZ_dT * self.temperature_span,
            dZ_dP * self.pressure_span,
        ], dtype=float)


class ThermoGPUOracle:
    """Keep ``thermogpu_tdar_oracle --stdin`` alive for a TDAR run.

    The executable must emit seven whitespace-separated fields per request::

        T_K P_Pa Z dZ_dT dZ_dP root_classification cubic_residual

    Only ``one_real`` states are accepted by default.  Values and physical
    derivatives are transformed into TDAR's normalized unit-square coordinates.
    """

    def __init__(
        self,
        executable: str | Path | Sequence[str],
        *,
        domain: ThermoGPUDomain = ThermoGPUDomain(),
        require_root: str | None = "one_real",
    ):
        self.domain = domain
        self.require_root = require_root
        if isinstance(executable, (str, Path)):
            command = [str(executable)]
        else:
            command = [str(x) for x in executable]
        if not command:
            raise ValueError("executable command cannot be empty")
        self.command = (*command, "--stdin")
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> "ThermoGPUOracle":
        if self._process is not None:
            return self
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    def __enter__(self) -> "ThermoGPUOracle":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __call__(self, uv):
        if self._process is None:
            self.start()
        process = self._process
        assert process is not None and process.stdin is not None and process.stdout is not None
        if process.poll() is not None:
            self._raise_process_failure(process, "ThermoGPU oracle exited before request")

        temperature, pressure = self.domain.physical(uv)
        try:
            process.stdin.write(f"{temperature:.17g} {pressure:.17g}\n")
            process.stdin.flush()
        except BrokenPipeError:
            self._raise_process_failure(process, "ThermoGPU oracle closed its input")

        line = process.stdout.readline()
        if not line:
            self._raise_process_failure(process, "ThermoGPU oracle produced no response")
        fields = line.split()
        if len(fields) != 7:
            raise RuntimeError(f"malformed ThermoGPU oracle response: {line.rstrip()!r}")
        try:
            returned_t, returned_p, z, dz_dt, dz_dp = map(float, fields[:5])
            residual = float(fields[6])
        except ValueError as exc:
            raise RuntimeError(f"non-numeric ThermoGPU oracle response: {line.rstrip()!r}") from exc
        root = fields[5]
        numeric = np.array([returned_t, returned_p, z, dz_dt, dz_dp, residual])
        if not np.all(np.isfinite(numeric)):
            raise RuntimeError(f"non-finite ThermoGPU oracle response: {line.rstrip()!r}")
        if not np.isclose(returned_t, temperature, rtol=0.0, atol=1e-9):
            raise RuntimeError("ThermoGPU oracle returned a mismatched temperature")
        if not np.isclose(returned_p, pressure, rtol=0.0, atol=1e-6):
            raise RuntimeError("ThermoGPU oracle returned a mismatched pressure")
        if self.require_root is not None and root != self.require_root:
            raise RuntimeError(
                f"ThermoGPU state T={temperature:g} K, P={pressure:g} Pa has root classification {root!r}; "
                f"required {self.require_root!r}"
            )
        return z, self.domain.normalized_gradient(dz_dt, dz_dp)

    @staticmethod
    def _raise_process_failure(process: subprocess.Popen[str], message: str):
        stderr = ""
        if process.stderr is not None and process.poll() is not None:
            stderr = process.stderr.read().strip()
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(message + detail)
