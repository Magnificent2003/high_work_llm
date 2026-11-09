import json, argparse, random, re
from pathlib import Path
from collections import Counter,defaultdict
from four_source_ablation_demo import validate_schema, exact_or_accuracy, FIELD_NAMES, pairwise_agreement, bootstrap_ci

KEYMAP={'harness_body_status':'harness_presence_status','hook_connection':'hook_attachment_status','hook_connection_status':'hook_attachment_status','harness_compliance':'harness_usage_status','harness_compliance_status':'harness_usage_status','harness_overall_status':'harness_usage_status','harness_use_status':'harness_usage_status','harness_overall_usage_status':'harness_usage_status','harness_overall_use_status':'harness_usage_status','overall_harness_use_status':'harness_usage_status','work_at_height_status':'work_at_height','height_work_status':'work_at_height','high_altitude_work_status':'work_at_height','working_at_height_status':'work_at_height','safety_helmet_status':'helmet_status','risk_level':'final_risk_level'}
VALMAP={
 'harness_presence_status':{'worn':'present','detected':'present','visible':'present','present':'present','yes':'present','unclear':'unclear'},
 'hook_attachment_status':{'unclear':'unclear','attached':'attached','connected':'attached','not_attached':'not_attached','disconnected':'not_attached','not_applicable':'not_applicable'},
 'harness_usage_status':{'unclear':'unclear','proper_use':'proper_use','improper_use':'improper_use','not_worn':'not_worn','not_applicable':'not_applicable'},
 'work_at_height':{'yes':'yes','no':'no','unclear':'unclear','confirmed':'yes',True:'yes',False:'no'},
 'helmet_status':{'worn':'worn','wearing':'worn','present':'worn','detected':'worn','yes':'worn','not_worn':'not_worn','unclear':'unclear'},
 'final_risk_level':{'low':'low','medium':'medium','high':'high','unclear':'unclear'} }

def repair(obj):
 if not isinstance(obj,dict): return None,[]
 out={}; notes=[]
 if 'overall_risk_level' in obj: out['overall_risk_level']=obj['overall_risk_level']
 elif 'overall_risk' in obj: out['overall_risk_level']=obj['overall_risk']; notes.append('key:overall_risk->overall_risk_level')
 if not isinstance(obj.get('persons'),list): return out,notes
 persons=[]
 for i,p in enumerate(obj['persons']):
  if not isinstance(p,dict): continue
  q={'person_id':str(p.get('person_id',i))}
  for k,v in p.items():
   nk=KEYMAP.get(k,k)
   if nk not in FIELD_NAMES[1:]: continue
   if nk!=k: notes.append(f'key:{k}->{nk}')
   if nk in VALMAP:
    
    try: nv=VALMAP[nk].get(v,VALMAP[nk].get(str(v)))
    except TypeError: nv=None
    if nv is None: continue
    if nv!=v: notes.append(f'value:{nk}:{v}->{nv}')
    q[nk]=nv
   else: q[nk]=v
  persons.append(q)
 out['persons']=persons
 return out,notes

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 rows=[]; unm=Counter(); goldbase=Path('experiments/runs/gold_draft_001/gold')
 for line in (a.run/'raw_results.jsonl').open(encoding='utf-8'):
  r=json.loads(line); fixed,notes=repair(r.get('parsed_output')); r['repaired_output']=fixed; r['repair_notes']=notes
  if fixed is not None:
   gold=goldbase/(r['sample_id']+'.json'); g=json.loads(gold.read_text(encoding='utf-8')) if gold.exists() else None
   r['repaired_schema']=validate_schema(fixed); r['repaired_gold_metrics']=exact_or_accuracy(fixed,g)
  else: r['repaired_schema']={'valid':False,'errors':['not_repairable']}; r['repaired_gold_metrics']={'gold_available':False}
  rows.append(r)
 (a.out/'rescored_rows.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
 summary={'n_rows':len(rows),'conditions':{},'mapping_policy':'Only explicit semantic aliases and value aliases; missing/ambiguous values remain missing and are never filled.'}
 for c in sorted(set(r['condition'] for r in rows)):
  sub=[r for r in rows if r['condition']==c]; valid=[r for r in sub if r['repaired_schema']['valid']]; gm=[r['repaired_gold_metrics'] for r in sub if r['repaired_gold_metrics'].get('gold_available')]
  fa={f:sum(m.get('field_accuracy',{}).get(f,0) for m in gm)/len(gm) if gm else None for f in FIELD_NAMES}
  exact=sum(m.get('exact_match',False) for m in gm)/len(gm) if gm else None
  groups=defaultdict(list)
  for r in sub: groups[r['sample_id']].append(r['repaired_output'])
  agr=[pairwise_agreement(v) for v in groups.values() if pairwise_agreement(v) is not None]
  summary['conditions'][c]={'n_samples':len(groups),'n_calls':len(sub),'json_parse_rate':sum(r.get('parsed_output') is not None for r in sub)/len(sub),'schema_valid_rate':len(valid)/len(sub),'exact_match_rate':exact,'field_accuracy':fa,'repeat_exact_agreement_clustered_by_image':sum(agr)/len(agr) if agr else None}
 full={r['sample_id']:r for r in rows if r['condition']=='full' and r['repeat']==0}
 paired={}
 for c in summary['conditions']:
  if c=='full': continue
  oth={r['sample_id']:r for r in rows if r['condition']==c and r['repeat']==0}; paired[c]={}
  for f in FIELD_NAMES:
   d=[full[s]['repaired_gold_metrics'].get('field_accuracy',{}).get(f,0)-oth[s]['repaired_gold_metrics'].get('field_accuracy',{}).get(f,0) for s in full if s in oth]
   if d: paired[c][f]={'mean_full_minus_ablated':sum(d)/len(d),'bootstrap_95_ci_image_clustered':bootstrap_ci(d),'n_pairs_images':len(d)}
 summary['paired_full_minus_ablated_image_clustered']=paired
 (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 (a.out/'comparison.csv').write_text('condition,valid_rate,field_accuracy_mean,exact_match_rate\n'+'\n'.join(f"{c},{v['schema_valid_rate']:.6f},{sum(x for x in v['field_accuracy'].values() if x is not None)/sum(x is not None for x in v['field_accuracy'].values()):.6f},{v['exact_match_rate']:.6f}" for c,v in summary['conditions'].items()),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()


