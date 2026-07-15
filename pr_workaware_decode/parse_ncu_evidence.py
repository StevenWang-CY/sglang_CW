#!/usr/bin/env python3
"""Re-derive Report-13 headline ncu numbers directly from the raw CSVs."""
import csv, statistics, os, sys, glob

BASE = "/Users/chuyuewang/Desktop/RESEARCH/Better Agentic Browser/sglang_log/offline_batch_results/offwall_profile"

DUR = "gpu__time_duration.avg"          # us (per captured launch)
DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"  # %
L2 = "lts__t_sector_hit_rate.pct"       # %
SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"      # %
GRID = "launch__grid_size"
WAVES = "launch__waves_per_multiprocessor"
RDSUM = "dram__bytes_op_read.sum"       # Mbyte
NKV = "device__attribute_multiprocessor_count"

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            kn = (r.get("Kernel Name") or "").strip()
            if not kn:
                continue
            rows.append(r)
    return rows

def fnum(r, k):
    v = (r.get(k) or "").replace(",", "").strip()
    try: return float(v)
    except: return None

def attn_rows(rows):
    out = []
    for r in rows:
        kn = r["Kernel Name"]
        if ("BatchPrefillWithPagedKVCacheKernel" in kn or
            "BatchDecodeWithPagedKVCacheKernel" in kn):
            out.append(r)
    return out

def summarize(path, label):
    if not os.path.exists(path):
        print(f"  [MISSING] {label}: {path}")
        return
    rows = load(path)
    ar = attn_rows(rows)
    if not ar:
        # show what kernels exist
        kns = sorted(set(r["Kernel Name"].split("<")[0].split("(")[0][:50] for r in rows))
        print(f"  [{label}] no attn kernel; kernels present: {kns[:6]}")
        return
    kind = "TENSOR-CORE(prefill-tmpl)" if "BatchPrefill" in ar[0]["Kernel Name"] else "CUDA-CORE(BatchDecode)"
    durs = [fnum(r, DUR) for r in ar if fnum(r, DUR) is not None]
    drams = [fnum(r, DRAM) for r in ar if fnum(r, DRAM) is not None]
    l2s = [fnum(r, L2) for r in ar if fnum(r, L2) is not None]
    grids = [int(fnum(r, GRID)) for r in ar if fnum(r, GRID) is not None]
    waves = [fnum(r, WAVES) for r in ar if fnum(r, WAVES) is not None]
    rd = [fnum(r, RDSUM) for r in ar if fnum(r, RDSUM) is not None]
    gridset = sorted(set(grids))
    nsm = fnum(ar[0], NKV)
    print(f"  [{label}] {kind}")
    print(f"     n_launches={len(ar)}  grids={gridset}  SM_count={nsm}")
    print(f"     dur_us: med={statistics.median(durs):.2f} min={min(durs):.2f} max={max(durs):.2f}")
    print(f"     DRAM%:  med={statistics.median(drams):.2f} min={min(drams):.2f} max={max(drams):.2f}")
    print(f"     L2hit%: med={statistics.median(l2s):.2f}")
    print(f"     waves/SM: med={statistics.median(waves):.3f}")
    print(f"     Sigma_read_MB={sum(rd):.1f}")
    return dict(med_dur=statistics.median(durs), med_dram=statistics.median(drams),
               grids=gridset, sum_rd=sum(rd), n=len(ar))

print("="*70)
print("ENGINE cells (ncu inside bench_one_batch) — main pass")
print("="*70)
summarize(f"{BASE}/bob_q25_3b_all_ps128_bs2_kv1024.csv", "Qwen2.5-3B (GQA-2 shape) B2/L1024 ps128  <- THE CELL")
summarize(f"{BASE}/bob_q25_3b_all_ps1_bs2_kv1024.csv",   "Qwen2.5-3B (GQA-2 shape) B2/L1024 ps1")
summarize(f"{BASE}/bob_q3vl_2b_all_ps128_bs2_kv1024.csv","Qwen3-VL-2B (GQA-8 shape) B2/L1024 ps128 (contrast)")
summarize(f"{BASE}/bob_q25_3b_all_ps128_bs8_kv4096.csv", "Qwen2.5-3B B8/L4096 ps128 (throughput ref)")

print()
print("="*70)
print("ENGINE cells — rep3 GUARANTEED-EXCLUSIVE pass")
print("="*70)
summarize(f"{BASE}/rep3/bob_q25_3b_all_ps128_bs2_kv1024.csv", "rep3 Qwen2.5-3B B2/L1024 ps128")
summarize(f"{BASE}/rep3/bob_q3vl_2b_all_ps128_bs2_kv1024.csv","rep3 Qwen3-VL-2B B2/L1024 ps128")

print()
print("="*70)
print("MICROBENCH arms at the constructed cell (single-launch wrappers)")
print("="*70)
summarize(f"{BASE}/flashinfer_distinct_all_h16k2_ps128_bs2_kv1024.csv", "GQA-2 CUDA-core microbench B2/L1024 ps128")
# tensor-core microbench arm
for cand in glob.glob(f"{BASE}/*h16k2tc*ps128*bs2*kv1024*.csv"):
    summarize(cand, f"GQA-2 tensor-core microbench {os.path.basename(cand)}")

print()
print("="*70)
print("DRAM% GRID (GQA-2 microbench CUDA-core, ps128 cold) — verify a few cells")
print("="*70)
for (b,l) in [(1,512),(1,1024),(1,2048),(2,512),(2,1024),(2,2048),(4,512),(8,4096)]:
    p = f"{BASE}/flashinfer_distinct_all_h16k2_ps128_bs{b}_kv{l}.csv"
    if os.path.exists(p):
        rows = attn_rows(load(p))
        if rows:
            d = [fnum(r,DRAM) for r in rows if fnum(r,DRAM) is not None]
            g = sorted(set(int(fnum(r,GRID)) for r in rows if fnum(r,GRID) is not None))
            print(f"  B{b}/L{l}: DRAM%={statistics.median(d):.1f}  grid={g}")
