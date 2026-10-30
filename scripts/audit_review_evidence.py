"""Read existing evidence, preserve provenance, and compute ranking counterfactuals. No model calls."""
import json,re,sys
from pathlib import Path
from collections import Counter
from datetime import datetime
from src.backend import db,services
from src.backend.graphs.search import persistent_graph,graph_config
from src.backend.text_cleaning import redact_contacts

def main():
    folder=Path('data/benchmarks/proof/review-followup');folder.mkdir(parents=True,exist_ok=True)
    db.init_db()
    with db.connect() as conn:
        failed=[dict(r) for r in conn.execute("SELECT id,error,created_at FROM cv WHERE status='failed'")]
    for row in failed:
        message=redact_contacts(row.pop('error') or 'unknown')[0]
        key=services.gemini_api_key()
        row.update(run_id=None,reason=message.replace(key,'[REDACTED]') if key else message,
                   provenance='legacy row: no original run_id column; do not infer membership from timestamps')
    source_id='a91025e0f20f489db58dda8d4a739efe'
    with persistent_graph() as graph:
        state=graph.get_state(graph_config(source_id)).values
    candidates={r['cv_id']:r['similarity'] for r in state.get('candidates',[])}
    criteria=state.get('criteria') or {}
    shortlist=db.get_shortlist(source_id)
    cases=[]
    for row in shortlist:
        if not any(x in row['cv_id'] for x in ('62c4d272','700fef76','2bc3301a')):continue
        cv=db.get_cv(row['cv_id']) or {}
        text=redact_contacts(cv.get('raw_text') or '')[0]
        excerpts=[m.group().strip() for m in re.finditer(r'[^.\n]{0,70}\b(?:years?|stud[yi]\w*)\b[^.\n]{0,120}',text,re.I)][:6]
        similarity=candidates.get(row['cv_id'])
        from src.backend.scoring import score_cv
        hypothetical=score_cv(row['cv_id'],{**cv,'years_experience':None},criteria,similarity)['score'] if similarity is not None else None
        reviewed='Under 3; exact years not established' if '62c4d272' in row['cv_id'] else 'Unknown professional years; study duration is not work experience' if '700fef76' in row['cv_id'] else 'Stored 2; source gives almost 2 years for one project plus a few years elsewhere; total professional years uncertain'
        counter_rank=1+sum(r['score']>hypothetical or (r['score']==hypothetical and r['cv_id']<row['cv_id']) for r in shortlist if r['cv_id']!=row['cv_id']) if hypothetical is not None else None
        cases.append({'cv_id':row['cv_id'],'source_excerpts':excerpts,'extracted_years':cv.get('years_experience'),
            'manual_assessment':reviewed,'rank':row['rank'],'score':row['score'],'similarity':similarity,
            'unknown_years_sensitivity':{'score':hypothetical,'rank_within_saved_top10':counter_rank,
                'scope':'Replace this CV years with unknown only; keep other nine saved scores fixed. Not a rerun or corrected ground truth.'}})
    log=Path('data/benchmarks/proof/live-validation-20261024/pilot.log').read_text(encoding='utf-8')
    lines=[x for x in log.splitlines() if 'AFC is enabled' in x or 'phase=extract_cache' in x or 'phase=embed_upsert' in x or 'batchEmbedContents' in x]
    base=datetime.strptime(lines[0][:23],'%Y-%m-%d %H:%M:%S,%f')
    timeline=[{'relative_seconds':round((datetime.strptime(x[:23],'%Y-%m-%d %H:%M:%S,%f')-base).total_seconds(),3),'event':x[24:]} for x in lines[:6]+lines[-5:]]
    runs=[]
    for run_id,path in [('pilot-A-20','data/benchmarks/gemini-dataset-run.json'),('pilot-B-20','data/benchmarks/proof/live-validation-20261024/pilot-run.json'),('pilot-B-resume','data/benchmarks/proof/live-validation-20261024/resume-run.json')]:
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        runs.append({'run_id':run_id,'id_origin':'Audit label assigned to existing artifact, not an original native run ID','source_artifact':path,
                     **{k:data.get(k) for k in ('requested','processed','ready','failed','already_ready','extract_sources','elapsed_seconds')}})
    report={'scope':'Existing evidence only; no corpus ingest or provider calls','runs':runs,'historical_failures':{'count':len(failed),'reason_counts':dict(Counter(r['reason'] for r in failed)),'records':failed},
            'source_search_id':source_id,'criteria':criteria,'extraction_cases':cases,'timeline':timeline}
    (folder/'evidence-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('runs','extraction_cases','timeline')},ensure_ascii=False))
    print(json.dumps(report['historical_failures']['reason_counts'],ensure_ascii=False))
    return 0

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');raise SystemExit(main())
