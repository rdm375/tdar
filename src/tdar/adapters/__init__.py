"""Optional adapters for derivative providers and external numerical oracles."""

from .thermogpu import ThermoGPUDomain, ThermoGPUOracle

__all__ = ["ThermoGPUDomain", "ThermoGPUOracle"]
