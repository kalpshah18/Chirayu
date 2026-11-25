"""Real NVML backend. Used at the pilot, on a real GPU; not exercised in the
finale, which runs entirely on ReplayBackend. Unit-tested with a mocked
`pynvml` so the interface is proven correct before any hardware is available
(IMPLEMENTATION PLAN.md §6.2).
"""
from __future__ import annotations

from typing import Optional

from . import DeviceInfo, Sample

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - expected in the finale environment
    pynvml = None


class NvmlBackend:
    def __init__(self):
        if pynvml is None:
            raise RuntimeError(
                "pynvml is not installed. NvmlBackend is for the pilot stage, "
                "on a machine with a real NVIDIA GPU."
            )
        pynvml.nvmlInit()
        self._handles: dict[str, "pynvml.c_nvmlDevice_t"] = {}

    def devices(self) -> list[DeviceInfo]:
        n = pynvml.nvmlDeviceGetCount()
        out = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            self._handles[uuid] = h
            out.append(DeviceInfo(id=uuid, sku=name, site="pilot"))
        return out

    def sample(self, dev: DeviceInfo) -> Sample:
        h = self._handles[dev.id]

        def safe(fn, *args):
            try:
                return fn(h, *args)
            except Exception:
                return None

        temp_core = safe(pynvml.nvmlDeviceGetTemperature, pynvml.NVML_TEMPERATURE_GPU)
        power_mw = safe(pynvml.nvmlDeviceGetPowerUsage)
        power_w = power_mw / 1000.0 if power_mw is not None else None
        energy_mj = safe(pynvml.nvmlDeviceGetTotalEnergyConsumption)
        util = safe(pynvml.nvmlDeviceGetUtilizationRates)
        util_pct = util.gpu if util is not None else None

        ecc = None
        try:
            ecc = pynvml.nvmlDeviceGetTotalEccErrors(
                h, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, pynvml.NVML_VOLATILE_ECC
            )
        except Exception:
            pass

        remapped = None
        try:
            remapped_info = pynvml.nvmlDeviceGetRemappedRows(h)
            remapped = remapped_info[0] if remapped_info else None
        except Exception:
            pass

        # HBM temperature and NVLink CRC errors go through nvmlDeviceGetFieldValues
        # in production; left as None here pending field-id wiring at the pilot.
        temp_mem: Optional[float] = None

        import datetime

        return Sample(
            t=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            temp_core_c=temp_core,
            temp_mem_c=temp_mem,
            power_w=power_w,
            energy_j=energy_mj / 1000.0 if energy_mj is not None else None,
            ecc_uncorrected=ecc,
            remapped_rows=remapped,
            xid=None,  # bound via an XID event callback in production, not polling
            utilization_pct=util_pct,
        )
