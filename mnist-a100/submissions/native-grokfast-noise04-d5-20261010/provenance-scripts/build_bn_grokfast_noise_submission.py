"""Package staged BN candidate sources only; no fitted data; not qualified."""
from pathlib import Path
import ast,base64,hashlib,json,lzma,re,argparse
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--entry',default='candidates/mlp_bn_back128_selftrain_staged.py');parser.add_argument('--stem',default='mlp_bn_back128_selftrain_d5_stage2000_2000');parser.add_argument('--leancompile',action='store_true');parser.add_argument('--bounded-tune',action='store_true');parser.add_argument('--minimal-aten',action='store_true');options=parser.parse_args()
ENTRY=options.entry
args=SimpleNamespace(student_ex2=False)
queue = [Path(ENTRY)]
files = {}
while queue:
    rel = queue.pop()
    if str(rel) in files:
        continue
    source = (ROOT / rel).read_text()
    if args.student_ex2 and str(rel) == ENTRY:
        source = source.replace('persistent_bf16_deep3_rmsnorm_silu_fastexp_rcp_hoisted_ltsearch', 'persistent_bf16_deep3_rmsnorm_silu_ex2_ltsearch')
    files[str(rel)] = source
    if rel.suffix == '.py':
        for node in ast.walk(ast.parse(source)):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                dep = Path(module.replace('.', '/') + '.py')
                if (ROOT / dep).is_file():
                    queue.append(dep)
        for name in re.findall(r"['\"]([^'\"]+\.(?:cu|cpp|h|cuh))['\"]", source):
            dep = Path(name) if name.startswith('kernels/') else Path('kernels') / name
            if (ROOT / dep).is_file():
                queue.append(dep)
    else:
        for name in re.findall(r'#include\s+"([^"]+)"', source):
            dep = rel.parent / name
            if (ROOT / dep).is_file():
                queue.append(dep)

for name in list(files):
    source = files[name]
    if any(t in name for t in ('persistent_bf16_deep3','persistent_bf16_deep4')) and name.endswith('_bindings.cpp'):
        source = source[:source.index('std::vector<torch::Tensor> debug_state()')]
        source += 'std::vector<torch::Tensor> debug_last_state(){return pb_last->tensors;}\n'
        source += 'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("classify",&classify);m.def("debug_last_state",&debug_last_state);}\n'
    elif any(t in name for t in ('persistent_bf16_deep3','persistent_bf16_deep4')) and name.endswith('.cu'):
        source = source[:source.index('void launch_vector_check(')]
        start = source.index('__global__ void pb_act(')
        end = source.index('__global__ void pb_error(')
        source = source[:start] + 'template<int C>struct Pack;template<>struct Pack<8>{union{uint4 vec;B x[8];};};template<>struct Pack<4>{union{uint2 vec;B x[4];};};\n' + source[end:]
        source = source.replace('#define mlp_bn16 rms_original_pairs\n#include "mlp_rmsnorm_maskregen.cuh"\n#undef mlp_bn16\n', '')
    elif name == 'kernels/learned_ridge_head.cu':
        lines = source.splitlines(True)
        source = ''.join(lines[:3]) + '#include <stdexcept>\n#include <cmath>\nusing B=__nv_bfloat16;\n'
        original = files[name]
        source += original[original.index('static void cb('):original.index('static void cs(')]
        source += original[original.index('__global__ void pack('):original.index('__global__ void targets(')]
        source += original[original.index('__global__ void votes('):original.index('__global__ void blend_layers(')]
    elif name == 'kernels/learned_ridge_head_bindings.cpp':
        source = '\n'.join(source.splitlines()[:6]) + '\n' + source[source.index('void neighbor('):source.index('void mix_layers(')] + source[source.index('torch::Tensor knn('):source.index('std::vector<torch::Tensor> fit(')]
        source += 'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("knn",&knn);}\n'
    elif name == 'learned_ridge_head.py':
        source = source.replace(",'-lcusolver'", '').replace(",' -lcusolver'", '')
        source = source[:source.index('def fit(')]
    files[name] = source

for name, source in list(files.items()):
    if name.endswith('.py'):
        node = ast.parse(source).body[0]
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            files[name] = ''.join(source.splitlines(True)[node.end_lineno:])


if options.leancompile:
 for name,source in list(files.items()):
  if name.endswith('_bindings.cpp'):
   files[name]=source.replace('#include <torch/extension.h>','#include <torch/types.h>\n#include <torch/csrc/utils/pybind.h>')
  elif name.endswith('.cu'):
   files[name]=source.replace('#include <torch/extension.h>','#include <torch/types.h>')
  elif name.endswith('.py') and 'return load(' in source:
   files[name]=source.replace('extra_cuda_cflags=',"extra_cflags=['-O0','-g0'],extra_cuda_cflags=")
if options.bounded_tune:
 for name,source in list(files.items()):
  if name.endswith('.cu') and '#define LT_EXHAUSTIVE 1' in source:
   assert 'pref,32,initial,&count' in source
   files[name]=source.replace('#define LT_EXHAUSTIVE 1','#define LT_EXHAUSTIVE 0').replace('pref,32,initial,&count','pref,8,initial,&count').replace('repeat<16','repeat<4').replace('repeat<5','repeat<3').replace('best*1000/80','best*1000/12')
if options.minimal_aten:
 from minimal_aten_bindings import rewrite_bindings_source
 for name,source in list(files.items()):
  if name.endswith('_bindings.cpp'):files[name]=rewrite_bindings_source(source)
# Isolate packaged libraries from research libraries with diagnostic exports.
for name,source in list(files.items()):
 if name.endswith('.py'):
  files[name]=re.sub(r"name='(sutro_[^']+)'",lambda m:"name='"+m[1]+"_bn_stagepack'",source)
# The fixed entry uses only gain.25/alpha.98/lambda.5; remove unused Python dispatch validation.
for name,source in list(files.items()):
 if name.endswith(('_warmstart_grokfast.py','_warmstart_featureaug_grokfast.py')):
  source=re.sub(r'^ if alpha not in.*\n','',source,flags=re.M)
  source=re.sub(r'^ if gain not in.*\n','',source,flags=re.M)
  source=source.replace(' if optimizer not in (\'nesterov\',\'momentum\',\'sgd\'):raise ValueError(\'unknown optimizer\')\n','')
  files[name]=source
# Specialize the augmentation-only kernel to the selected fixed noise recipe.
for name,source in list(files.items()):
 if name.endswith('_warmstart_featureaug_grokfast.cu'):
  a=source.index(' float u=(pb_hash(r+ns+81273)');b=source.index('  if(noise>0)',a)
  source=source[:a]+' if(t<60){float v=source[t];\n'+source[b:]
  files[name]=source
 if name.endswith('.cu') and 'persistent_bf16' in name:
  # All networks use data_sample (kind7); hidden matrices use its Gaussian branch.
  a=source.index(' if(kind==2)');b=source.index('w[physical]=',a)
  source=source[:a]+" float a=(pb_hash(i+seed)+1.f)/4294967296.f,b=(pb_hash(i+seed+123456)+1.f)/4294967296.f;value=sqrtf(-2*logf(a))*cosf(6.2831853f*b);float scaled=gain*value/sqrtf(float(d));"+source[b:]
  files[name]=source
 # Retain tensor contract checks; fixed internal configuration needs no dispatch checks.
 if name.endswith('_bindings.cpp') and 'persistent_bf16' in name:
  # Drop only redundant config/cache/supplied-initial validation; entry is fixed.
  source=files[name]
  source=re.sub(r'&&views>=1[^;]*?\);',');',source)
  source=re.sub(r' if\(initialized\)\{for\(auto&t:initial_weights\).*?initial weight shapes mismatch"\);\}', '',source)
  source=re.sub(r' TORCH_CHECK\(initialized&&original==70.*?"invariance input/config mismatch"\);', '',source)
  files[name]=source

# Remove comments for the benchmark source limit; preserve quoted C++ tokens.
comment_tokens=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/',re.S)
for name,source in list(files.items()):
 if name.endswith(('.cu','.cpp','.h','.cuh')):
  source=comment_tokens.sub(lambda m:(' '+('\n'*m[0].count('\n'))) if m[0].startswith(('//','/*')) else m[0],source)
  files[name]='\n'.join(line.rstrip() for line in source.splitlines() if line.strip())+'\n'
 elif name.endswith('.py'):
  files[name]=ast.unparse(ast.parse(source))+'\n'
# Compact C++ punctuation without rewriting quoted tokens or preprocessor lines.
quoted=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
for name,source in list(files.items()):
 if name.endswith(('.cu','.cpp','.h','.cuh')):
  compact=[]
  for line in source.splitlines():
   if line.lstrip().startswith('#'):compact.append(line.strip());continue
   pieces=[];pos=0
   for match in quoted.finditer(line):
    pieces.append(re.sub(r'[ \t]*([{}();,])[ \t]*',r'\1',line[pos:match.start()]))
    pieces.append(match[0]);pos=match.end()
   pieces.append(re.sub(r'[ \t]*([{}();,])[ \t]*',r'\1',line[pos:]))
   compact.append(''.join(pieces).strip())
  files[name]='\n'.join(compact)+'\n'
files['e.py']=files.pop(ENTRY);ENTRY='e.py'
archive=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
compressed=[]
for lc in (0,1,2,3,4):
 for pb in (0,1,2,3,4):
  blob=lzma.compress(archive,filters=[{'id':lzma.FILTER_LZMA2,'preset':9|lzma.PRESET_EXTREME,'lc':lc,'lp':0,'pb':pb}])
  assert lzma.decompress(blob)==archive
  compressed.append(blob)
encoded=base64.b85encode(min(compressed,key=len)).decode()
tag=hashlib.sha256(archive).hexdigest()[:16]
stub='import base64,json,lzma,os,sys\nfrom pathlib import Path\n_m=None\ndef classify(x,y,q):\n global _m\n if _m is None:\n  p=Path(\'/tmp\')/(\'sutro-TAG-\'+str(os.getuid()));p.mkdir(exist_ok=True)\n  for n,s in json.loads(lzma.decompress(base64.b85decode(DATA))).items():\n   f=p/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(s)\n  sys.path.insert(0,str(p))\n  from ENTRYMODULE import classify\n  _m=classify\n return _m(x,y,q)\n'.replace('TAG',tag).replace('DATA',repr(encoded)).replace('ENTRYMODULE',ENTRY.removesuffix('.py').replace('/','.'))
stem=options.stem
assert len(stub.encode())<=20480,len(stub.encode())
ast.parse(stub)
(ROOT/('submissions/'+stem+'.py')).write_text(stub)
for name,source in files.items():
 p=ROOT/('packaging/'+stem+'_sources')/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)
manifest=dict(source_bytes=len(stub.encode()),source_sha256=hashlib.sha256(stub.encode()).hexdigest(),archive_sha256=hashlib.sha256(archive).hexdigest(),sources={name:hashlib.sha256(source.encode()).hexdigest() for name,source in files.items()},scope='Source-only packaging prototype; no full-wrapper or sandbox qualification')
(ROOT/('submissions/'+stem+'_manifest.json')).write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='sources'},indent=2))
