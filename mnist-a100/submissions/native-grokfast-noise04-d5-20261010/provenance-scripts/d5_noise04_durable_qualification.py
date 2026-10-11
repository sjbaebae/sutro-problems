"""Canonical run_modal scorer with detached recoverable call handles; no scorer changes."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import modal
from run_modal import app,remote_score
import mnist
SOURCE=ROOT/'submissions/mlp_bn_head16_d5_warmstart_grokfast_noise04.py'
RESULTS=ROOT/'results/modal-d5-warmstart-grokfast-noise04-disk-recovery-20261010'
MANIFEST=RESULTS/'launch-manifest.json'
def save(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:
  json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)
p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','collect'));args=p.parse_args()
RESULTS.mkdir(exist_ok=True)
if args.action=='launch':
 if MANIFEST.exists():raise RuntimeError('Already launched; collect existing calls instead')
 raw=SOURCE.read_bytes();mnist.check_source(raw,'classify',SOURCE.name)
 info={'source':str(SOURCE),'source_sha256':hashlib.sha256(raw).hexdigest(),'source_bytes':len(raw),'difficulty':5,'sandbox':'required','calls':[],'runner':'unchanged run_modal.remote_score, three independent single-use A100 containers','prior_attempt':'../modal-d5-warmstart-grokfast-noise04-initial-20261010/infrastructure-audit.json','scope':'Explicitly authorized recovery batch after disk failure; prior unknown outcomes disclosed, no claim that this was first-ever attempt.'}
 with modal.enable_output(),app.run(detach=True):
  info['app_id']=app.app_id;save(MANIFEST,info)
  for i in range(3):
   call=remote_score.spawn(SOURCE.name,raw.decode(),'classify',['--difficulty','5'])
   info['calls'].append({'run':i+1,'function_call_id':call.object_id});save(MANIFEST,info)
 print(json.dumps(info),flush=True)
else:
 info=json.loads(MANIFEST.read_text());assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==info['source_sha256']
 assert len(info['calls'])==3
 pending=[v for v in info['calls'] if not (RESULTS/f"run-{v['run']}.json").exists()]
 while pending:
  for item in list(pending):
   try:value=modal.FunctionCall.from_id(item['function_call_id']).get(timeout=0)
   except (TimeoutError,modal.exception.TimeoutError):continue
   except Exception as e:
    save(RESULTS/f"collection-error-{item['run']}.json",{'exception':type(e).__name__,'message':str(e),'call':item});raise
   save(RESULTS/f"run-{item['run']}.json",value);pending.remove(item)
   record=value.get('record') or {}
   print(json.dumps({'run':item['run'],'returncode':value['returncode'],'summary':record.get('summary'),'problems':record.get('problems'),'ranked_ms':record.get('ranked_ms')}),flush=True)
  if pending:time.sleep(20)
 print('All three original recovery-batch outputs saved.',flush=True)
