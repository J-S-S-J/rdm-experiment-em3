"""
analyze_ddm.py — Quick DDM-ready data check & summary
=======================================================
Run after collecting data to verify your CSV is well-formed and to
generate a summary table suitable for DDM fitting with HDDM or PyDDM.

    python src/analyze_ddm.py results/PARTICIPANT_timestamp.csv

Outputs
-------
- Console summary: accuracy & median RT per coherence level
- results/<participant>_summary.csv : condition-level summary
- results/<participant>_ddm_ready.csv : cleaned trial-level file for fitting

DDM fitting notes
-----------------
The trial-level CSV has exactly the columns expected by HDDM:
    subj_idx, response, rt, coherence, accuracy

Install HDDM:  pip install hddm  (requires Python ≤ 3.9 + PyMC2)
Install PyDDM: pip install pyddm (pure-Python, easier to install)

Example HDDM snippet (run separately after this script):

    import hddm
    data = hddm.load_csv('results/<participant>_ddm_ready.csv')
    m = hddm.HDDM(data, depends_on={'v': 'coherence'})
    m.find_starting_values()
    m.sample(2000, burn=500)
    m.print_stats()
"""

import sys
import os
import csv
import math
import statistics


def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/analyze_ddm.py results/<file>.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    rows = load_csv(input_path)

    # Filter to main (non-practice) trials with valid RTs
    main_rows = [
        r for r in rows
        if r.get('is_practice', '1') == '0'
        and safe_float(r.get('reaction_time')) is not None
        and safe_float(r.get('reaction_time', 'nan')) == safe_float(r.get('reaction_time', 'nan'))  # not NaN
    ]

    print(f"\n{'='*55}")
    print(f"  RDM Data Summary")
    print(f"  File: {os.path.basename(input_path)}")
    print(f"  Total main trials: {len(main_rows)}")
    print(f"{'='*55}")

    # Group by coherence
    coherences = sorted(set(r['coherence'] for r in main_rows),
                        key=lambda x: float(x))

    summary_rows = []
    print(f"\n{'Coherence':>12} {'N':>5} {'Accuracy':>10} {'Median RT':>12} {'Mean RT':>12}")
    print('-' * 55)

    for coh in coherences:
        coh_rows = [r for r in main_rows if r['coherence'] == coh]
        n = len(coh_rows)
        accs  = [int(r['accuracy']) for r in coh_rows if r['accuracy'] != '']
        rts   = [safe_float(r['reaction_time']) for r in coh_rows
                 if safe_float(r['reaction_time']) is not None]

        acc_mean = statistics.mean(accs) if accs else float('nan')
        rt_med   = statistics.median(rts) if rts else float('nan')
        rt_mean  = statistics.mean(rts)   if rts else float('nan')

        print(f"{float(coh):>12.2f} {n:>5} {acc_mean:>10.3f} "
              f"{rt_med:>12.4f} {rt_mean:>12.4f}")

        summary_rows.append({
            'coherence': coh,
            'n_trials': n,
            'accuracy': round(acc_mean, 4),
            'median_rt': round(rt_med, 4),
            'mean_rt': round(rt_mean, 4),
        })

    # Write summary CSV
    base = os.path.splitext(input_path)[0]
    summary_path = base + '_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['coherence', 'n_trials', 'accuracy',
                           'median_rt', 'mean_rt'])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved → {summary_path}")

    # Write HDDM-ready CSV
    # HDDM expects: subj_idx, response (0/1), rt (seconds), + regressors
    ddm_path = base + '_ddm_ready.csv'
    pid = main_rows[0].get('participant_id', 'P01') if main_rows else 'P01'

    with open(ddm_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['subj_idx', 'rt', 'response', 'accuracy', 'coherence',
                      'direction', 'trial_number']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in main_rows:
            rt = safe_float(r.get('reaction_time'))
            if rt is None or rt <= 0:
                continue   # skip missing / non-positive RTs

            # HDDM convention: response=1 for correct, 0 for error
            acc = r.get('accuracy', '')
            response_hddm = int(acc) if acc in ('0', '1') else ''

            writer.writerow({
                'subj_idx':    pid,
                'rt':          round(rt, 6),
                'response':    response_hddm,
                'accuracy':    acc,
                'coherence':   r.get('coherence', ''),
                'direction':   r.get('direction', ''),
                'trial_number': r.get('trial_number', ''),
            })

    print(f"DDM-ready file saved → {ddm_path}")
    print(f"\nTo fit a DDM, load '{os.path.basename(ddm_path)}' "
          f"into HDDM or PyDDM.\n")


if __name__ == '__main__':
    main()
