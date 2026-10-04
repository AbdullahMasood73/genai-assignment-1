"""Measure CPU tensor layouts on disposable models; no dataset/checkpoint changes."""
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from restoration.models import Autoencoder, reconstruction


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    original = Autoencoder(base=24,latent=8192,dropout=0,detail=True,
                           detail_bn=True,detail_shuffle=True)
    image = torch.rand(16,3,128,128)
    target = torch.rand_like(image)
    normal = original.eval()(image).detach()
    alternative = copy.deepcopy(original).to(memory_format=torch.channels_last).eval()
    candidate = alternative(image.to(memory_format=torch.channels_last)).detach()
    max_difference = float((normal-candidate).abs().max())
    torch.testing.assert_close(normal,candidate,atol=1e-5,rtol=1e-4)
    records = {}
    for name,layout in [('contiguous',torch.contiguous_format),('channels_last',torch.channels_last)]:
        model = copy.deepcopy(original).to(memory_format=layout).train()
        x,y = image.to(memory_format=layout), target.to(memory_format=layout)
        optimizer = torch.optim.Adam(model.parameters(),lr=.0003)
        times=[]
        for step in range(12):
            started=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss=reconstruction(model(x),y,.8)
            loss.backward()
            optimizer.step()
            if step>=3:
                times.append(time.perf_counter()-started)
        model.eval()
        validation=[]
        with torch.no_grad():
            for step in range(12):
                started=time.perf_counter()
                loss=reconstruction(model(x),y,.8)
                if step>=3:
                    validation.append(time.perf_counter()-started)
        records[name]={'train_median_seconds':float(np.median(times)),
                       'validation_median_seconds':float(np.median(validation))}
        print(name,records[name],flush=True)
    result={'scope':'Disposable random inputs/models; not assignment training or quality results',
            'torch':torch.__version__,'threads':4,'batch_size':16,
            'output_max_absolute_difference':max_difference,'measurements':records}
    (BASE/'experiments/cpu_layout_benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
