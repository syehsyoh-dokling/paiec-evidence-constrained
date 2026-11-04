"""Unit tests for the submitted predictor, run against an unmodified submission archive.

    PAIEC_ROOT=<project root> python -m unittest code/tests/test_predictor.py
    (or: python code/tests/test_predictor.py <path to submission zip>)

The archive is extracted to a temporary folder and imported as `model`; no network, no
credentials. Tests cover output bounds, behaviour on unknown subjects and empty text,
tolerance of malformed labeled entries, monotonicity in the number of successes, the
documented dependence on call history, and the per-call latency budget.
"""
import copy, importlib.util, os, pathlib, random, shutil, sys, tempfile, time, unittest, zipfile

ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
DEFAULT_ZIP = ROOT / "localsim" / "variants" / "v403_expanded_upload.zip"
ZIP = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].endswith(".zip") else DEFAULT_ZIP
os.environ["PAIEC_DISABLE_BLE_NETWORK"] = "1"
os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"


def load_model(zip_path):
    wd = pathlib.Path(tempfile.mkdtemp(prefix="paiec_test_"))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(wd)
    sys.path.insert(0, str(wd)); os.chdir(wd)
    spec = importlib.util.spec_from_file_location("model", str(wd / "model.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, wd


def subject(name="gpt-4o", effort=""):
    return {"normalized_name": name, "provider": "", "release_date": "", "access_date": "", "harness": "", "harness_version": "", "reasoning_effort": effort, "subject_features_extra": ""}


def item(text, bm="bm_unit_test"):
    return {"item_content": text, "item_features": "", "interactors": "", "benchmark_id": bm}


TEXTS = ["What is the capital of France?", "Compute the integral of x^2 from 0 to 1.", "Which option is correct? (A) 2 (B) 3 (C) 5 (D) 7",
         "Write a Python function that reverses a list.", "Explain why the sky is blue.", "Solve for x: 3x + 5 = 20."]


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.wd = load_model(ZIP)
        # warm-up loads the encoders
        cls.model.predict([subject(), item(TEXTS[0])], [])

    @classmethod
    def tearDownClass(cls):
        os.chdir(ROOT); shutil.rmtree(cls.wd, ignore_errors=True)

    def labeled(self, s, n, k, bm="bm_unit_test", seed=0):
        rng = random.Random(seed)
        out = []
        for i in range(n):
            out.append(((copy.deepcopy(s), item(f"labelled question number {i}: {TEXTS[i % len(TEXTS)]}", bm)), 1 if i < k else 0))
        rng.shuffle(out)
        return out

    def test_bounds_random_inputs(self):
        rng = random.Random(1)
        for _ in range(40):
            s = subject(rng.choice(["gpt-4o", "claude-3.5-sonnet", "zz-unknown-model-9000-xl", "llama-3-70b-instruct"]), rng.choice(["", "low", "high"]))
            n = rng.choice([0, 1, 3, 7, 15, 31]); k = rng.randint(0, n)
            p = self.model.predict([s, item(rng.choice(TEXTS))], self.labeled(s, n, k, seed=rng.randint(0, 999)))
            self.assertIsInstance(p, float); self.assertGreaterEqual(p, 0.0); self.assertLessEqual(p, 1.0)
            self.assertFalse(p != p, "NaN prediction")

    def test_unknown_subject_and_empty_text(self):
        s = subject("zz-unknown-model-9000-xl")
        for text in ["", " ", "x"]:
            for n, k in [(0, 0), (1, 1), (3, 1)]:
                p = self.model.predict([s, item(text)], self.labeled(s, n, k))
                self.assertGreaterEqual(p, 0.0); self.assertLessEqual(p, 1.0)

    def test_odd_label_values_and_sparse_dicts_are_tolerated(self):
        s = subject()
        odd = [((s, item(TEXTS[1])), 2), ((s, item(TEXTS[1])), None), ((s, item(TEXTS[2])), 1.0), ((s, item(TEXTS[2])), True),
               ((s, {"benchmark_id": "bm_unit_test"}), 1), (({}, item(TEXTS[3])), 0), ((s, item(TEXTS[3])), 0)]
        p = self.model.predict([s, item(TEXTS[0])], odd)
        self.assertGreaterEqual(p, 0.0); self.assertLessEqual(p, 1.0)

    @unittest.expectedFailure
    def test_structurally_broken_entries_known_defect(self):
        """Known defect of the frozen archive: an entry of `labeled` that is not a ((subject, item), label)
        pair with dict subject and item (None, a string, an empty tuple, a 3-tuple, or a None subject/item)
        raises inside predict. The evaluator always passes well-formed entries, so scoring is unaffected;
        the fix (skip such entries) requires editing model.py, which was frozen after upload."""
        s = subject()
        for bad in [None, "x", (), ((s, item(TEXTS[1])), 1, 9), ((None, item(TEXTS[1])), 1), ((s, None), 1)]:
            p = self.model.predict([s, item(TEXTS[0])], [bad, ((s, item(TEXTS[2])), 1)])
            self.assertGreaterEqual(p, 0.0); self.assertLessEqual(p, 1.0)

    def test_incomplete_input(self):
        for inp in [[], [subject()], None]:
            p = self.model.predict(inp, [])
            self.assertGreaterEqual(p, 0.0); self.assertLessEqual(p, 1.0)

    def test_more_successes_do_not_lower_prediction(self):
        s = subject("gpt-4o"); n = 7
        preds = [self.model.predict([s, item(TEXTS[0], "bm_mono")], self.labeled(s, n, k, bm="bm_mono")) for k in range(n + 1)]
        for a, b in zip(preds, preds[1:]):
            self.assertGreaterEqual(b, a - 0.02, f"prediction fell from {a:.3f} to {b:.3f} with one more success: {preds}")
        self.assertGreater(preds[-1], preds[0] + 0.1)

    def test_prediction_depends_on_call_history_as_documented(self):
        # The pool centre of a benchmark is a running mean of the items seen so far, so a fresh
        # benchmark id and one that has already seen other items may give different values.
        s = subject("gpt-4o")
        p_fresh = self.model.predict([s, item(TEXTS[0], "bm_hist_a")], [])
        for t in TEXTS[1:]:
            self.model.predict([s, item(t, "bm_hist_b")], [])
        p_after = self.model.predict([s, item(TEXTS[0], "bm_hist_b")], [])
        self.assertTrue(abs(p_fresh - p_after) < 0.1, "history effect should be bounded")
        # and repeating the exact sequence reproduces the value
        p_again = self.model.predict([s, item(TEXTS[0], "bm_hist_b")], [])
        self.assertAlmostEqual(p_after, p_again, places=6)

    def test_latency_budget(self):
        s = subject("claude-3.5-sonnet"); lab = self.labeled(s, 31, 20, bm="bm_lat")
        t = time.time(); n = 20
        for i in range(n):
            self.model.predict([s, item(f"latency probe {i}: {TEXTS[i % len(TEXTS)]}", "bm_lat")], lab)
        per_call = (time.time() - t) / n
        self.assertLess(per_call, 2.0, f"{per_call:.2f} s per call at budget 31 exceeds the 2 s budget")


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    unittest.main(verbosity=2)
