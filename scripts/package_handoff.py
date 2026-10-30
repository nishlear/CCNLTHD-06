"""Package committed source and selected pilot evidence; never includes credentials or whole corpus."""
import hashlib,json,sqlite3,subprocess,zipfile
from pathlib import Path
from src.backend import config,services

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/benchmarks/proof/review-followup'

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding="utf-8").strip()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sample=OUT/'handoff-checkpoints.db'
    thread='13dbc9cc7fad4f29a96bdf686fd3d9e5'
    with sqlite3.connect(config.CHECKPOINT_DB_PATH) as source,sqlite3.connect(sample) as target:
        source.execute('BEGIN')
        for name,ddl in source.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name IN ('checkpoints','writes')"):
            exists=target.execute('SELECT 1 FROM sqlite_master WHERE type=? AND name=?',('table',name)).fetchone()
            if not exists:target.execute(ddl)
            columns=[row[1] for row in source.execute(f'PRAGMA table_info({name})')]
            if 'thread_id' not in columns:raise ValueError('Unexpected checkpoint schema')
            rows=source.execute(f'SELECT * FROM {name} WHERE thread_id=?',(thread,)).fetchall()
            if rows:target.executemany(f'INSERT OR REPLACE INTO {name} VALUES ({",".join("?" for _ in columns)})',rows)
        target.commit()
    paths=git('ls-files','--cached','--others','--exclude-standard').splitlines()
    files={f'source/{name}':ROOT/name for name in paths if name and not name.startswith('data/') and Path(name).name!='.env'}
    selected=['data/benchmarks/gemini-dataset-run.json','data/benchmarks/gemini-model-check.json',
        'data/datasets/djinni/sample-100000.manifest.json',
        'data/benchmarks/environment_20261024.json',
        'data/benchmarks/djinni_100k_filtered_20261024.json',
        'data/benchmarks/djinni_100k_fts.json',
        'data/benchmarks/experiments_20261024.json',
        'data/benchmarks/gemini-flash-lite-check.json',
        'data/benchmarks/report-audit-20261024.json',
        'data/benchmarks/pilot_100_20261024.json',
        'data/benchmarks/gemini-execution-20261024.json',
        'data/benchmarks/gemini-deadline-investigation-20261024.json',
        'data/benchmarks/proof/chapter12-editorial/report-before.pdf',
        'data/benchmarks/proof/chapter12-editorial/chapter12-before.md',
        'data/benchmarks/proof/chapter12-editorial/editorial-verification.json',
        'data/benchmarks/proof/chapter12-editorial/doi-chieu-before.md',
        'data/benchmarks/pilot-20-results-20261024.json',
        'data/benchmarks/pilot-20-verification-20261024.json',
        'data/benchmarks/proof/live-validation-20261024/pilot-input.jsonl',
        'data/benchmarks/proof/live-validation-20261024/pilot-input.manifest.json',
        'data/benchmarks/proof/live-validation-20261024/pilot-run.json',
        'data/benchmarks/proof/live-validation-20261024/pilot.log',
        'data/benchmarks/proof/live-validation-20261024/resume-run.json',
        'data/benchmarks/proof/live-validation-20261024/before-review.json',
        'data/benchmarks/proof/live-validation-20261024/after-review.json',
        'data/benchmarks/proof/live-validation-20261024/qwen-real-extract.json',
        'data/benchmarks/proof/live-validation-20261024/summary.json']
    image_folder=ROOT/'data/benchmarks/proof/report-audit-20261024'
    if image_folder.is_dir():
        selected += [p.relative_to(ROOT).as_posix() for p in image_folder.iterdir()
                     if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')]
    selected += [str(p.relative_to(ROOT)).replace('\\','/') for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.xml','.log','.db','.pdf') and p.name!='handoff-manifest.json' and p.name!='lab-checkpoints.db']
    for name in selected:
        if (ROOT/name).is_file():files['evidence/'+name]=ROOT/name
    dirty=bool(git('status','--porcelain'))
    if dirty:
        patch=OUT/'working-tree.patch';patch.write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT))
        files['working-tree.patch']=patch
    manifest={'git_head':git('rev-parse','HEAD'),'working_tree_dirty':dirty,
        'head_scope':'Committed source only; archived file hashes and patch describe any uncommitted changes',
        'pilot_limit':20,'models':services.model_settings(),'checkpoint_thread_id':thread,
        'scope':'Source, pilot B input 20, selected historical evidence, checkpoint sample and deterministic lab; no new provider calls',
        'files':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}}
    key=services.gemini_api_key()
    if key:
        for name,path in files.items():
            if key.encode() in path.read_bytes():raise ValueError('Credential scan failed for '+name)
    (OUT/'handoff-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=OUT.with_name('jd-matching-handoff.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name,path in files.items():bundle.write(path,arcname=name)
        bundle.write(OUT/'handoff-manifest.json',arcname='manifest.json')
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for name,digest in manifest['files'].items():assert hashlib.sha256(bundle.read(name)).hexdigest()==digest
    print(json.dumps({'archive':str(archive),'git_head':manifest['git_head'],'dirty':dirty,'files':len(files),'hashes_verified':True}))
    return 0

if __name__=='__main__':raise SystemExit(main())
