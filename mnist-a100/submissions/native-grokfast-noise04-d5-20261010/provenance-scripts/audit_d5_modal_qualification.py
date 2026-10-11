"""Audit complete canonical records; never substitute or discard failed runs."""
import argparse, hashlib, json, statistics, sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mnist

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--results', type=Path, required=True)
args = parser.parse_args()
sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
runs = []
for number in (1, 2, 3):
    path = args.results / f'run-{number}.json'
    if not path.exists():
        runs.append(dict(run=number, status='PENDING'))
        continue
    raw = json.loads(path.read_text())
    record = raw.get('record') or {}
    calls = record.get('calls', [])
    checks = dict(source=record.get('source_sha256') == sha,
                  sandbox=record.get('sandboxed') is True,
                  band=record.get('band_bp') == 190,
                  complete=len(calls) == 15 and sum(c['dataset'] == 'mnist' for c in calls) == 11)
    judged = None
    if checks['complete']:
        problems, ranked, summary = mnist.judge([SimpleNamespace(**c) for c in calls], 190)
        judged = dict(problems=problems, ranked_ms=ranked, summary=summary)
        # Canonical scorer clears the ranked field when judge finds a failure.
        checks['judge_matches'] = problems == record.get('problems') and (None if problems else ranked) == record.get('ranked_ms')
        checks['accuracy_and_protocol'] = not problems
    else:
        checks['accuracy_and_protocol'] = False
    energy = record.get('energy') or {}
    checks['energy'] = energy.get('mj_per_call') is not None and not energy.get('problems', ['missing energy'])
    checks['returncode'] = raw.get('returncode') == 0
    group = [c for c in calls if c['dataset'] == 'mnist']
    total = sum(c['total'] for c in group)
    runs.append(dict(run=number, status='PASS' if all(checks.values()) else 'FAIL',
                     checks=checks, error_pct=100 * (1 - sum(c['correct'] for c in group) / total) if total else None,
                     calls=len(calls), ranked_ms=record.get('ranked_ms'),
                     energy_mj=energy.get('mj_per_call'), gpu_uuid=energy.get('device', {}).get('uuid'),
                     original_problems=record.get('problems'), independent_judge=judged))
qualified = all(r['status'] == 'PASS' for r in runs)
uuids = [r.get('gpu_uuid') for r in runs]
distinct_devices = len(set(uuids)) == 3 and all(uuids)
qualified = qualified and distinct_devices
report = dict(source=str(args.source.resolve()), source_sha256=sha, runs=runs,
              three_distinct_devices=bool(distinct_devices), qualified=bool(qualified),
              median_ms=statistics.median(r['ranked_ms'] for r in runs) if qualified else None,
              median_mj=statistics.median(r['energy_mj'] for r in runs) if qualified else None,
              scope='All initial three records retained; no qualifying median unless every independent run passes.')
args.results.mkdir(parents=True, exist_ok=True)
(args.results / 'qualification-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
