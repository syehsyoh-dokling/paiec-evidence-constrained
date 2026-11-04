# Kontrak Produksi 70 Submisi oleh Qwen (Generator) dengan Claude sebagai QA/Reviewer

Tanggal: 2026-11-04. Berlaku untuk `agent_loop.py` + `local_llm.py` (Qwen2.5-Coder di T4).
Pembagian peran: **Qwen membuat**, **kode menegakkan gate**, **Claude hanya menilai** hasil
gate dan memberi verdict. Claude tidak menulis atau memperbaiki kandidat.

## 0. Fakta yang membatasi target (baca sebelum menulis prompt)

1. Tidak ada prediktor skor hosted. Noise skor formatif ±0.027 per submisi; skor ekspektasi
   keluarga v153 ≈ 0.185; 0.1276 adalah draw terbaik dari 172. Target "prediksi skor <0.12xxx"
   **tidak bisa dipenuhi oleh agen mana pun** dan tidak boleh dijadikan kriteria terima.
2. Kriteria terima yang bisa diukur: **ΔALC LOBO berpasangan vs v153** pada simulator
   `localsim/` (112 pair, 7 fold). Ekspektasi hosted ≈ 0.185 + ΔALC. Perbaikan budget-0
   terbaik sejauh ini: ΔALC = −0.0011 (SE 0.0005).
3. Throughput gate: diff hanya di blok budget-0 → eval cepat ~8 menit/kandidat (7 fold);
   diff di budget ≥1 → simulasi penuh ~12 menit/kandidat (4 worker). 70 kandidat = 10–14
   jam GPU. Kuota upload 50/hari, 36 sisa hari ini sudah dipakai 14, jadi 70 submisi =
   2 hari kuota. Deadline final 6 Nov 23:59 AoE.
4. Qwen2.5-Coder-1.5B gagal menjaga `old` persis; gunakan **7B-Instruct** dengan
   `temperature=0.2`, `max_new_tokens=400`.

## 1. System prompt untuk Qwen (verbatim, bahasa Inggris)

```text
You are a code-editing agent for the PAIEC predictor. You will be given ONE live code
snippet from model.py (lines with exact text) and ONE edit slot. Your only output is a
single <CHANGE> block:
<CHANGE>{"old": "<exact text copied from the snippet>", "new": "<replacement>", "hypothesis": "<one sentence, testable>", "slot": "<slot name>", "params": {<name>: <number>}}</CHANGE>
Hard rules (violating any one = your output is discarded, no retry credit):
1. "old" must be copied character-for-character from the snippet and occur exactly once.
2. Edit only inside the given slot. Never touch: def predict signature, imports, labeling.py,
   submission_config.json, data/, anything outside the snippet.
3. No network, no file writes, no new imports, no API keys, no benchmark_###### or
   subject_###### identifiers, no comments that mention hidden Codabench IDs or scores.
4. Numbers must stay inside the slot's allowed ranges below. A change of a single constant
   by less than 1% of its range is NOT a candidate.
5. Each candidate must differ from every earlier candidate in this batch in at least one
   parameter value (the batch list is supplied). Do not repeat a rejected candidate.
6. Return nothing else: no prose, no markdown, no explanation outside "hypothesis".
```

## 2. Slot yang diizinkan (design space, ditegakkan oleh `qa_gate.py` G4)

| Slot | Anchor di v153 `model.py` | Parameter & rentang | Alasan (bukti LOBO) |
|---|---|---|---|
| `B0` | blok `if not labeled:` (baris 954–962) | `center ∈ [0.48, 0.56]`, `shrink ∈ [0, 1]`, `ease ∈ [0, 1.2]`, cap bawah/atas `[0.02, 0.98]` | B0 v153 lebih buruk dari konstanta 0.5; fix −0.013 Brier B0 |
| `LOW_BUDGET_SHRINK` | blok `if count > 0:` … `p_base = clip(...)` (baris 1120–1143) | bobot blend `count==1 ∈ [0.20, 0.60]`, `count<=3 ∈ [0.05, 0.35]`, batas `count<=K, K ∈ {3,5,7}`, `debias ∈ [0.20, 0.32]` | B1 (0.2497) hampir = B0 (0.2559); 40 % bobot metrik di B1+B3 |
| `MIXTURE_PRIOR` | `_mixture_bayes_posterior` (baris 129–161) | `(a,b)` tiap regime, `a,b ∈ [0.3, 8.0]`, bobot `w1,w3 ∈ [0.10, 0.40]` | pseudo-count menentukan overreaksi pada 1–3 label |
| `PEER_SHIFT` | `prior_shift += 0.45 * (peer_bm_mean - 0.386) * n/(n+2)` (baris 1112–1118) | `0.45 → [0.25, 0.70]`, `0.386 → [0.35, 0.45]`, `2.0 → [1.0, 6.0]` | peer satu-satunya sinyal pada B1 |
| `IRT_ENVELOPE` | `env = clip(p(1-p)/0.06, 0.15, 1)` & `p_final = p_base + env*0.32*d_rel` (baris 1166–1170) | `0.06 ∈ [0.04, 0.12]`, `0.15 ∈ [0.05, 0.40]`, `0.32 ∈ [0.0, 0.50]` | B31 (0.1947) = oracle konstanta pair (0.1905): IRT tidak menambah diskriminasi |

Prompt per kandidat (dibangun `agent_loop.py`, bukan oleh Qwen): slot terpilih (round-robin
dengan kuota 20/20/10/10/10), snippet live ±4 baris dari anchor, daftar `params` kandidat
batch yang sudah ada, dan daftar `params` yang REJECT beserta alasannya.

## 3. Gate otomatis (kode, bukan LLM): `localsim/qa_gate.py`

| Gate | Isi | Hasil gagal |
|---|---|---|
| G1 | `py_compile` | REJECT |
| G2 | daftar entri ZIP identik dengan v153 | REJECT |
| G3 | tidak ada `api_keys`, string rahasia, import jaringan | REJECT |
| G4 | semua baris yang berubah berada di dalam slot yang diizinkan; hanya `model.py` yang berubah | REJECT |
| G5 | nama ≤40 karakter; SHA-256 unik vs batch dan ledger upload | REJECT |
| G6 | tidak ada ID benchmark/subjek hidden yang ditambahkan | REJECT |
| G7 | smoke `predict()` dengan 0, 1, 2 label ∈ [0,1] | REJECT |
| LOBO | B0-only → eval cepat 905 pair; selain itu → simulasi penuh 112 pair berpasangan vs v153 | lihat verdict |

Verdict:
* **PASS_UPLOAD**: ΔALC ≤ −0.0005, ΔALC + 2·SE < 0, dan ≥4 dari 7 fold membaik.
* **HOLD**: tidak signifikan (interval mencakup 0). Boleh diunggah hanya jika kuota tersisa
  setelah semua PASS, sebagai pengukur noise.
* **REJECT**: gate statis gagal, atau ΔALC − 2·SE > 0 (terbukti lebih buruk).

Perintah reviewer (dijalankan Claude, read-only terhadap kandidat):

```bash
python localsim/qa_gate.py <folder_kandidat> --out localsim/results/qa_<batch> --jobs 4
```

Keluaran: `qa_<nama>.json` per kandidat dan `qa_summary.json`; verdict dicetak per baris.

## 4. Alur batch 70

1. `agent_loop.py` menghasilkan kandidat ke `qwen_batch_<tgl>/cand_NNN.zip` beserta
   `cand_NNN.meta.json` (slot, params, hypothesis, hash). Tidak ada eval di dalam loop Qwen;
   loop hanya memaketkan.
2. Setelah ≥10 kandidat terkumpul, reviewer menjalankan `qa_gate.py` pada folder itu.
   Kandidat B0-only diproses dulu (murah); kandidat slot lain antre simulasi penuh.
3. Hasil REJECT dikembalikan ke prompt Qwen sebagai daftar larangan (params + alasan).
4. Upload hanya PASS_UPLOAD, diurutkan ΔALC paling negatif, lewat
   `localsim/upload_17_quota.py` yang diarahkan ke manifest batch (ledger otomatis).
5. Setiap hari sertakan 1 kontrol v153 identik dan 2 replikat dari PASS terbaik agar
   noise terukur berpasangan per `benchmark_id`.

## 4b. Strategi 5 langkah (usulan 4 Nov) dan implementasinya

| Langkah | Implementasi | Catatan QA |
|---|---|---|
| 1. Semua skor masuk ke pengetahuan Qwen | `build_qwen_knowledge_v2.py` → `qwen_knowledge_v2.json`: 172 skor hosted + 14 pending, statistik noise (sd per submisi, replikat ZIP identik), tabel 7 benchmark hidden dari log formatif, hasil LOBO (kurva penuh dan grid B0), definisi slot + rentang + nilai baseline | Disuntikkan utuh ke konteks Qwen sebelum prompt per kandidat |
| 2. Qwen memilih baseline terbaik | Qwen wajib menjawab `{"baseline": ..., "reason": ...}` sebelum kandidat pertama. **Jawaban diterima hanya jika alasannya memakai LOBO ΔALC dan garis keturunan kode**, bukan skor hosted terendah | Jika Qwen memilih "karena 0.1276 terendah", QA menandai *selection on noise* dan meminta ulang. Hasil yang benar tetap v153, karena v153 unggul 0.012 ALC vs baseline panitia di LOBO dan semua keluarga varian turun darinya |
| 3. Jadikan baseline | `localsim/variants/v153_nokeys.zip` (API key sudah dihapus) adalah satu-satunya sumber; setiap kandidat dibuat dari ZIP ini, bukan dari kandidat lain | Ditegakkan G2/G4 |
| 4. Reproduksi identik sebagai kontrol | 1 salinan byte-identik per hari upload, nama `..._control.zip`, tag `control` di meta | `qa_gate.py` menolak model.py identik; kontrol dikecualikan hanya lewat tag `control` dan dibatasi 1 per hari |
| 5. Distribusi naik-turun per kategori dan radius | Kategori = 5 slot. Radius dinyatakan sebagai **fraksi rentang slot**, bukan langkah absolut: r1 = ±25 %, r2 = ±50 %, r3 = seluruh rentang. Nilai disampel Latin-hypercube per slot (bukan ±0.0005 seperti `concentric_grid.py`, yang menghasilkan replika). Alokasi 70: kontrol 1, replikat PASS terbaik 2, B0 15, LOW_BUDGET_SHRINK 20, MIXTURE_PRIOR 12, PEER_SHIFT 10, IRT_ENVELOPE 10 | Setiap kandidat ≥1 % rentang dari semua kandidat lain; distribusi disimpan di `cand_NNN.meta.json` |

Koreksi penting pada langkah 2 dan 5: grid radius lama (`cb0` 0.4465…0.4500, `cpeer` 0.3840…0.3870)
menghasilkan 50 kandidat yang secara statistik identik; LOBO menunjukkan efek maksimalnya
≈0.001 ALC, sedangkan noise hosted 0.027. Radius harus cukup lebar untuk mengubah perilaku
(itulah rentang slot di §2), dan pemilihannya harus lewat LOBO, bukan skor hosted.

## 5. Yang tidak boleh ada di laporan atau prompt

* Klaim "prediksi skor hosted" untuk kandidat mana pun.
* Konstanta yang dipilih karena cocok dengan ID benchmark hidden di log formatif.
* Kandidat yang hanya mengganti komentar/nama file untuk mendapat SHA baru.
