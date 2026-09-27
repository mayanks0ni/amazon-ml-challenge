"""
Run the Business Entity Resolution pipeline end to end.

    python src/run_pipeline.py --data-dir data

Every step saves its results in --work-dir, so if the run stops you can resume with:
    python src/run_pipeline.py --data-dir data --steps from:features
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ber_lib as B  # noqa: E402

STEPS = ['normalize', 'baseline', 'candidates', 'features', 'stage2a', 'stage2b',
         'decide', 'errors', 'test', 'package']


def pick_steps(spec):
    if spec == 'all':
        return STEPS
    if spec.startswith('from:'):
        return STEPS[STEPS.index(spec[5:]):]
    out = [s.strip() for s in spec.split(',') if s.strip()]
    bad = [s for s in out if s not in STEPS]
    if bad:
        raise SystemExit(f'unknown step(s) {bad}; choose from {STEPS}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='folder with the 7 competition TSV files')
    ap.add_argument('--work-dir', default='work', help='intermediate files (normalized data, models, ...)')
    ap.add_argument('--out-dir', default='output', help='final matching_results.tsv / candidate_pairs.tsv')
    ap.add_argument('--steps', default='all', help=f'"all", "from:<step>" or a comma list of {STEPS}')
    ap.add_argument('--sample', type=int, default=150_000, help='train S1 entities used for training')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--low-memory', action='store_true', help='smaller chunks for machines with <= 16 GB RAM')
    ap.add_argument('--team-name', default='team')
    ap.add_argument('--force', action='store_true', help='redo normalization / test scoring even if files exist')
    a = ap.parse_args()

    cfg = dict(train_s1_sample=a.sample, n_folds=a.folds, workers=a.workers)
    if a.low_memory:
        cfg.update(max_pairs_block=4_000_000, feat_chunk=500_000)
    cfg = B.cfg_with(cfg)
    os.makedirs(a.work_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    res_path = os.path.join(a.work_dir, 'results.json')
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    def save():
        json.dump(results, open(res_path, 'w'), indent=1, default=str)

    steps = pick_steps(a.steps)
    B.log(f'steps: {steps}   config: {cfg}')
    t0 = time.time()
    for step in steps:
        B.log(f'================ {step.upper()} ================')
        if step == 'normalize':
            B.step2_normalize(a.data_dir, a.work_dir, cfg, force=a.force)
        elif step == 'baseline':
            f, rep = B.step3_baseline(a.data_dir, a.work_dir, a.out_dir, cfg)
            results['baseline'] = {'train_macro_f05': f, 'validation': rep}
        elif step == 'candidates':
            results['candidate_recall'] = B.step4_train_candidates(a.data_dir, a.work_dir, cfg)
        elif step == 'features':
            B.step5_train_features(a.work_dir, cfg)
        elif step == 'stage2a':
            results['stage2a'] = B.step6_train_stage2a(a.work_dir, cfg)
        elif step == 'stage2b':
            results['stage2b'] = B.step7_train_stage2b(a.work_dir, cfg)
        elif step == 'decide':
            results['decision'] = B.step8_decide(a.work_dir, cfg)
        elif step == 'errors':
            results['error_analysis'] = B.step9_error_analysis(a.work_dir, cfg)
        elif step == 'test':
            results['test_validation'] = B.step11_predict_test(a.data_dir, a.work_dir, a.out_dir, cfg, force=a.force)
        elif step == 'package':
            zip_path = os.path.abspath(f'{a.team_name}_submission.zip')
            B.package_submission(a.work_dir, a.out_dir, os.path.join(HERE, 'ber_lib.py'), zip_path, results, cfg)
            results['package'] = zip_path
        save()
    B.log(f'done in {(time.time() - t0) / 60:.1f} min; summary saved to {res_path}')


if __name__ == '__main__':      # required on Windows/macOS for multiprocessing
    main()
