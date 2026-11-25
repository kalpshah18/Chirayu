# Physical witness — pilot hardware design

Not built or demonstrated at the finale. The finale substitutes Summit's own
power-supply input-power reading as the independent channel (see
`replay_witness.py` and IMPLEMENTATION PLAN.md §6.3). This document is the
design for the pilot, where a real lessor's GPU needs a real physical witness.

## Components

- **ESP32-S3** microcontroller, running the witness firmware.
- **PZEM-004T v3** AC power meter module (voltage, current, active power,
  energy, frequency, power factor) on the node's AC input.
- **CT clamp** sized to the node's PDU circuit rating.
- **Sealed enclosure** with a tamper switch wired to a GPIO interrupt: any
  attempt to open the enclosure or disconnect the CT clamp is logged and
  signed as a `WITNESS_TAMPER` event before the witness goes silent.
- **Its own Ed25519 key**, generated on first boot inside the enclosure and
  never exported, so a compromised lessee machine cannot forge witness
  records even with full control of the host OS.

## Safety rule

**CT clamp only on lab-checked wiring.** Never install on live wiring without
a qualified electrician verifying the circuit, insulation rating and clamp
sizing first. This is a hard rule, not a guideline — a wrongly sized or
installed CT clamp is a fire and shock hazard.

## Data path

1. PZEM-004T samples AC input power at ~1 Hz over its UART interface.
2. ESP32-S3 batches readings every 10 s, signs the batch with its witness
   key, and sends it over Wi-Fi/Ethernet to the ledger's `/v1/witness/batch`
   endpoint — the same endpoint the finale's `replay_witness.py` posts to.
3. The ledger treats the witness stream exactly as it treats Summit's PSU
   channel: an independent measurement to regress the agent's self-reported
   power against (`ledger/divergence.py`).

## Bill of materials (indicative, pilot-scale, per node)

| Item | Approx. cost (INR) |
| --- | --- |
| ESP32-S3 dev board | 600 |
| PZEM-004T v3 + CT clamp (rated to PDU circuit) | 1,200 |
| Enclosure + tamper switch | 400 |
| **Total per node** | **~2,200** |

## Optional stage prop

A much smaller version — a metering smart plug under a laptop, reading CPU
package power via Linux RAPL as the "self-reported" side — can demonstrate
the same divergence check live on stage without any lab wiring. This is a
stretch goal (IMPLEMENTATION PLAN.md §6.3), not required for the finale.
