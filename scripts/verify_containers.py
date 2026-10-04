"""Exercise the real Compose HTTP application with uploaded images."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import httpx
from PIL import Image

BASE = Path(__file__).resolve().parents[1]


def image_check(value):
    assert value.startswith('data:image/png;base64,')
    with Image.open(BytesIO(base64.b64decode(value.split(',',1)[1]))) as image:
        assert image.size == (128,128) and image.mode == 'RGB'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8080')
    parser.add_argument('--face-image',type=Path,required=True)
    args = parser.parse_args()
    pet = (BASE/'backend/samples/sample-1.png').read_bytes()
    face = args.face_image.read_bytes()
    records = []
    with httpx.Client(base_url=args.url,timeout=60) as client:
        health = client.get('/api/health').raise_for_status().json()
        assert health['status']=='ready' and all(health['workspaces'].values())
        assert '<html' in client.get('/').raise_for_status().text
        for corruption in ('salt','blur','occlusion'):
            outputs = []
            for _ in range(2):
                row = client.post('/api/corrupt',files={'file':('pet.png',pet,'image/png')},
                    data={'corruption':corruption,'severity':'medium','seed':'42'}).raise_for_status().json()
                image_check(row['output'])
                outputs.append(row['output'])
            assert outputs[0]==outputs[1],corruption
            records.append({'operation':'corrupt','condition':corruption,'reproducible_seed':True})
        operations = [('universal-restoration',pet,{}),('hard-routing',pet,{}),('soft-mixture',pet,{})]
        operations += [('face-to-sketch',face,{'style':str(style)}) for style in range(3)]
        for endpoint, photo, fields in operations:
            row = client.post('/api/'+endpoint,files={'file':('input.png',photo,'image/png')},
                data={'corruption':'salt' if endpoint!='face-to-sketch' else 'none',
                      'severity':'medium','seed':'42',**fields}).raise_for_status().json()
            image_check(row['output'])
            for key in ('weights','probabilities'):
                if key in row:
                    values = row[key].values() if isinstance(row[key],dict) else row[key]
                    assert abs(sum(values)-1)<1e-5
            records.append({'operation':endpoint,'style':fields.get('style'),
                            'status':200,'inference_ms':row['inference_ms']})
        experiments = client.get('/api/experiments').raise_for_status().json()
        assert experiments['pet_input_count']==36690 and experiments['face_pair_count']==1046
        assert experiments['tracking']['runs']
    result = {'status':'passed','url':args.url,'health':health,'operations':records,
              'mlflow_runs_visible':len(experiments['tracking']['runs']),
              'scope':'Real container HTTP integration and PNG validity; not a quality or throughput benchmark.'}
    path = BASE/'docs/container-verification.json'
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
