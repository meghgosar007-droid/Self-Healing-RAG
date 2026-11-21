import json

import pytest

from eval import dataset


def test_shipped_dataset_is_valid_and_has_a_contract_smoke_subset():
    qs = dataset.load_questions()
    dataset.validate_smoke_subset(qs)
    assert all(q.id and q.question and q.reference_notes for q in qs)
    assert any(q.answerable for q in qs) and any(not q.answerable for q in qs)
    full = dataset.select(qs, "full")
    smoke = dataset.select(qs, "smoke")
    assert len(full) == len(qs) and 0 < len(smoke) < len(full)
    assert all(q.smoke for q in smoke)


def test_dataset_rows_have_exactly_the_contract_fields():
    lines = dataset.DATASET_PATH.read_text(encoding="utf-8").splitlines()
    for line in lines:
        assert set(json.loads(line)) == {"id", "question", "answerable", "smoke", "reference_notes"}


def q(i, answerable, smoke=True):
    return dataset.Question(
        id=f"q{i}", question="?", answerable=answerable, smoke=smoke, reference_notes="n"
    )


@pytest.mark.parametrize(
    "qs",
    [
        [q(i, True) for i in range(10)],  # too few total and no unanswerable
        [q(i, i % 2 == 0) for i in range(30)],  # too many smoke
        [q(i, i < 12) for i in range(15)],  # only 3 unanswerable
    ],
)
def test_smoke_validation_rejects_bad_subsets(qs):
    with pytest.raises(ValueError):
        dataset.validate_smoke_subset(qs)


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "q.jsonl"
    row = {"id": "q1", "question": "?", "answerable": True, "smoke": False, "reference_notes": ""}
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dataset.load_questions(p)
