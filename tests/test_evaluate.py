import csv
import json

from sil.evaluate import ModeScore, evaluate, log_test_run, report
from sil.metrics import Config
from sil.wildchat import LABEL_COLUMNS


def make_eval_dir(tmp_path, rows, split):
    d = tmp_path / "eval"
    (d / "data").mkdir(parents=True)
    for r in rows:
        payload = {"session_id": r["id"], "turns": [
            {"role": "user", "text": f"first message of {r['id']}"},
            {"role": "assistant", "text": "reply"},
        ]}
        (d / "data" / f"{r['id']}.json").write_text(json.dumps(payload), encoding="utf-8")
    with open(d / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LABEL_COLUMNS})
    (d / "split.json").write_text(json.dumps({"seed": 0, "split": split}), encoding="utf-8")
    return str(d)


def test_mode_score_math():
    s = ModeScore(tp=3, fp=1, fn=3, tn=10)
    assert s.precision == 0.75 and s.recall == 0.5
    assert abs(s.f1 - 0.6) < 1e-9
    assert ModeScore().precision is None and ModeScore().f1 is None


def test_evaluate_scores_only_the_requested_half(tmp_path):
    rows = [
        {"id": "a", "goal": "g", "validation_loop": "1", "premature_convergence": "0", "drift": "0"},
        {"id": "b", "goal": "", "validation_loop": "0", "premature_convergence": "0", "drift": "1"},
        {"id": "c", "goal": "g", "validation_loop": "0", "premature_convergence": "?", "drift": "0"},
        {"id": "d", "goal": "g"},                                  # not labeled yet
        {"id": "e", "goal": "g", "validation_loop": "1", "premature_convergence": "0", "drift": "0"},
    ]
    split = {"a": "calibration", "b": "calibration", "c": "calibration",
             "d": "calibration", "e": "test"}
    d = make_eval_dir(tmp_path, rows, split)

    predictions = {"a": {"validation_loop"}, "b": set(), "c": {"drift"}}
    seen_goals = {}

    def fake_detect(t, goal, cfg):
        seen_goals[t.session_id] = goal
        return predictions[t.session_id]

    r = evaluate(d, "calibration", Config(), detect=fake_detect)
    assert r.n_chats == 3
    assert (r.modes["validation_loop"].tp, r.modes["validation_loop"].tn) == (1, 2)
    assert (r.modes["drift"].fn, r.modes["drift"].fp) == (1, 1)
    assert r.modes["premature_convergence"].skipped == 1
    assert r.goals_from_first_message == 1
    assert seen_goals["b"] == "first message of b"
    assert "e" not in seen_goals
    assert "validation_loop" in report(r, verbose=True)


def test_test_runs_are_logged(tmp_path):
    rows = [{"id": "e", "goal": "g", "validation_loop": "1", "premature_convergence": "0", "drift": "0"}]
    d = make_eval_dir(tmp_path, rows, {"e": "test"})
    r = evaluate(d, "test", Config(), detect=lambda t, g, c: {"validation_loop"})
    assert log_test_run(d, r, Config()) == 1
    assert log_test_run(d, r, Config()) == 2
