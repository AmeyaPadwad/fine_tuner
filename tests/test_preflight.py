from src.train.preflight import check_hardware, estimate_vram_gb
from src.train.schema import JobSpec


def _job(tmp_path, **training_overrides) -> JobSpec:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}")
    payload = {
        "job_id": "job_vram_test",
        "dataset": {"source_path": str(data_path)},
        "training": training_overrides,
    }
    return JobSpec.model_validate(payload)


def test_laya_421m_estimate_fits_a_16gb_card(tmp_path):
    # Laya's base English checkpoint is ~421M params (ModernBERT-large).
    num_params = 421_000_000
    job = _job(tmp_path)
    estimate = estimate_vram_gb(num_params, job)
    # Full fine-tune of a 421M model should comfortably fit a single T4/16GB
    # card — this is the whole point of the reference run using 2xT4s.
    assert estimate.total_gb < 16.0


def test_larger_batch_and_seq_length_increase_estimate(tmp_path):
    num_params = 421_000_000
    small = _job(tmp_path, batch_size=1)
    large = _job(tmp_path, batch_size=32)
    assert estimate_vram_gb(num_params, large).total_gb > estimate_vram_gb(num_params, small).total_gb


def test_optimizer_state_dominates_over_model_weights(tmp_path):
    # Full AdamW fine-tuning: 2 fp32 moment buffers (8 bytes/param) vs 4
    # bytes/param for fp32 weights themselves — optimizer state should be
    # the larger term, unlike the old LoRA estimate where it was tiny.
    num_params = 421_000_000
    job = _job(tmp_path)
    estimate = estimate_vram_gb(num_params, job)
    assert estimate.optimizer_state_gb > estimate.model_weights_gb


def test_check_hardware_degrades_gracefully_without_torch():
    report = check_hardware()
    assert isinstance(report.summary(), str)
    if not report.torch_available:
        assert "torch" in report.summary().lower()
