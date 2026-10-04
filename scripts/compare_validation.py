"""Compare restoration candidates using complete validation measurements only."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    old,new = [json.loads(path.read_text()) for path in (args.baseline,args.candidate)]
    assert all(s['split']=='validation' and not s['subset'] for s in (old,new))
    assert old['pet_input_count']==new['pet_input_count']
    baseline={(r['system'],r['corruption'],r['severity']):r for r in old['restoration']}
    rows=[]
    for row in new['restoration']:
        key=(row['system'],row['corruption'],row['severity'])
        previous=baseline[key]
        rows.append({**{k:row[k] for k in ('system','corruption','severity','n')},
            'previous_l1':previous['l1'],'candidate_l1':row['l1'],
            'psnr_change_db':row['psnr']-previous['psnr'],
            'ssim_change':row['ssim']-previous['ssim'],
            'previous_objective':.8*previous['l1']+.2*(1-previous['ssim']),
            'candidate_objective':.8*row['l1']+.2*(1-row['ssim'])})
    systems={}
    for system in ('universal','oracle_routing','predicted_routing','soft'):
        group=[r for r in rows if r['system']==system]
        n=sum(r['n'] for r in group)
        before=sum(r['previous_objective']*r['n'] for r in group)/n
        after=sum(r['candidate_objective']*r['n'] for r in group)/n
        systems[system]={'previous_objective':before,'candidate_objective':after,
                         'improved':after<before}
    face=[]
    old_face={r['style']:r for r in old['face']}
    for row in new['face']:
        previous=old_face[row['style']]
        assert previous['n']==row['n']
        face.append({'style':row['style'],'n':row['n'],
            'previous_objective':.8*previous['l1']+.2*(1-previous['ssim']),
            'candidate_objective':.8*row['l1']+.2*(1-row['ssim']),
            'psnr_change_db':row['psnr']-previous['psnr'],
            'ssim_change':row['ssim']-previous['ssim']})
    n=sum(r['n'] for r in face)
    before=sum(r['previous_objective']*r['n'] for r in face)/n
    after=sum(r['candidate_objective']*r['n'] for r in face)/n
    systems['gan']={'previous_objective':before,'candidate_objective':after,'improved':after<before}
    result={'split':'validation','full_split':True,'systems':systems,'conditions':rows,'face_styles':face,
        'previous_test_already_observed':True,
        'interpretation':'Lower objective is better. Inspect per-condition changes and image grids before promotion.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(systems,indent=2))


if __name__ == '__main__':
    main()
