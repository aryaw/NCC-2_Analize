## Psudocode breakdown

## Intensity-based features
---
ByteRatio = TotBytes / (TotPkts + 1)
# Detects small-command packets or bulk-data exfiltration
# Normal user traffic tends to have a balanced relation between bytes and packets
# Botnet traffic often transmits many very small packets (commands, checks, heartbeats), producing a low ByteRatio
# Some malware exfiltrates data in large chunks > high ByteRatio

DurationRate = TotPkts / (Dur + 0.1)
# Detects packet-burst behavior and timing regularity typical of bots
# C&C communication often shows rapid bursts of packets, especially during check-ins or command distribution
# Normal traffic has more variability; botnet traffic has unusual consistency
# Lots of packets sent in a short time → typical of bots responding automatically

Intensity = TotBytes / (Dur + 1)
# Detects throughput anomalies and continuous C&C channels
# High intensity may indicate data exfiltration, bot updates, compressed payloads
Low intensity with long duration may indicate persistent background C&C channels (eg, periodic # keep-alive signals)

Why use division by +1 or +0.1?
# To avoid division by zero
    If TotPkts = 0, ByteRatio would break
    If Dur = 0, DurationRate & Intensity would break

---

## Directional features

PktByteRatio = TotPkts / (TotBytes + 1)
# Detects small control packets used in botnet C&C signaling
# High value → many small packets (typical for bot commands or probe packets)
# Low value → large data transfers (exfiltration, update delivery)

SrcByteRatio = SrcBytes / (TotBytes + 1)
# Reveals who is talking more, exposing C&C leadership characteristics
    If C&C server > bot, malicious commands flow mostly from the destination > low SrcByteRatio
    If bot > C&C, status reports flow from the source > high SrcByteRatio
# Behaviour:
    Bot receiving commands > low SrcByteRatio
    Bot sending logs/exfiltration > high SrcByteRatio

TrafficBalance = |sTos – dTos|
# Detects non-human packet structure differences, helping highlight bot-generated flows
# Large imbalance abnormal or machine-generated traffic
# Near-zero normal balanced flows

---

## Timing-Based Features

DurationPerPkt = Dur / (TotPkts + 1)
# Detects timing regularity, heartbeat signals, and burst behavior
# Low DurationPerPkt many packets sent very quickly
    common for bots receiving commands
    also seen in scanning or distributed probing
# High DurationPerPkt > slow, spaced-out communication
    typical of sleep cycles in botnets
    indicates periodic check-ins ("phone home" behavior)

FlowIntensity = SrcBytes / (TotBytes + 1)
# Captures directional timing rhythms (send/receive phases) of bot behavior
# Bots sending periodic status updates > high FlowIntensity
# Bots receiving commands from C&C > low FlowIntensity
# Bots alternating between command and report phases > fluctuating intensity cycles