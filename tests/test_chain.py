from agent.backends import DeviceInfo, Sample
from agent.chain import ChainBuilder, SigningKey, verify_chain


def _make_chain(n=20):
    key = SigningKey.generate()
    dev = DeviceInfo(id="test:gpu0", sku="V100")
    builder = ChainBuilder(dev, key)
    records = []
    for i in range(n):
        sample = Sample(t=f"2020-01-01T00:00:{i:02d}Z", temp_core_c=40.0 + i, temp_mem_c=44.0, power_w=200.0, xid=None)
        records.append(builder.record(sample, nonce=f"n{i}"))
    return key, dev, records


def test_clean_chain_verifies():
    key, _, records = _make_chain()
    problems = verify_chain(records, key.public_hex())
    assert problems == []


def test_single_field_edit_detected():
    key, _, records = _make_chain()
    records[5]["s"]["power_w"] = 9999.0  # tamper with a reading, signature now invalid
    problems = verify_chain(records, key.public_hex())
    assert any("signature" in p for p in problems)


def test_deletion_detected():
    key, _, records = _make_chain()
    del records[5]
    problems = verify_chain(records, key.public_hex())
    assert any("seq" in p or "prev" in p for p in problems)


def test_replayed_batch_detected():
    key, _, records = _make_chain()
    # Splice an earlier record back in later, as if someone replayed a batch.
    records[10] = dict(records[3])
    problems = verify_chain(records, key.public_hex())
    assert len(problems) > 0


def test_wrong_key_rejected():
    key, _, records = _make_chain()
    other_key = SigningKey.generate()
    problems = verify_chain(records, other_key.public_hex())
    assert all("signature" in p for p in problems if "record 0" in p)
    assert len(problems) > 0
