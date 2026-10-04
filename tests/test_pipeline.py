"""Behavioral checks use synthetic fixtures only; they are not assignment results."""
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient

from restoration.corruptions import apply, config, CLASSES
from restoration.data import BalancedBatches, Faces
from restoration.models import Autoencoder, Classifier, Generator, Discriminator, SoftMixture, ssim, reconstruction
from restoration.prepare import write_json, safe_extract

torch.set_num_threads(2)


def test_refined_gan_style_condition_and_onnx(tmp_path):
    from restoration.export import export_one
    torch.manual_seed(42)
    cfg={'base':16,'embedding':8,'dropout':0.0,'gan_refined':True}
    generator=Generator(**cfg).eval()
    discriminator=Discriminator(**cfg).eval()
    image=torch.rand(2,3,128,128)
    styles=torch.tensor([0,1])
    output=generator(image,styles)
    assert output.shape==image.shape
    assert discriminator(image,output,styles).shape[0]==2
    assert not torch.allclose(generator(image[:1],torch.tensor([0])),
                              generator(image[:1],torch.tensor([2])))
    checkpoint=tmp_path/'gan.pt'
    torch.save({'task':'gan','config':cfg,'model':generator.state_dict(),'synthetic':True,
                'epoch':0,'best':1.0,'manifest_sha256':'synthetic-fixture'},checkpoint)
    record=export_one(checkpoint,tmp_path/'models',permit_synthetic=True)
    assert record['verified']


def test_subpixel_initialization_has_matching_phases():
    from restoration.models import subpixel_conv
    conv=subpixel_conv(4,3)
    output=torch.nn.functional.pixel_shuffle(conv(torch.rand(1,4,8,8)),2)
    assert torch.equal(output[:,:,::2,::2],output[:,:,1::2,::2])
    assert torch.equal(output[:,:,::2,::2],output[:,:,::2,1::2])


@pytest.mark.parametrize('batch_norm,shuffle', [(False,False),(True,False),(True,True)])
def test_detail_autoencoder_compresses_and_exports_without_bypass(tmp_path, batch_norm, shuffle):
    import onnxruntime as ort
    torch.manual_seed(42)
    model = Autoencoder(base=16, latent=4096, detail=True, detail_bn=batch_norm, detail_shuffle=shuffle).eval()
    image = torch.rand(1, 3, 128, 128)
    compressed = model.compress(model.encoder(image))
    assert compressed.numel() == 4096 < image.numel()
    expected = model(image)
    model.compress.weight.data.zero_()
    model.compress.bias.data.zero_()
    assert torch.allclose(model(image), model(torch.zeros_like(image)), atol=1e-6)
    # Restore the learned path before checking the actual inference export.
    model = Autoencoder(base=16, latent=4096, detail=True, detail_bn=batch_norm, detail_shuffle=shuffle).eval()
    expected = model(image).detach().numpy()
    destination = tmp_path / 'detail.onnx'
    torch.onnx.export(model, image, destination, input_names=['image'], output_names=['output'],
                      opset_version=17, dynamo=False)
    actual = ort.InferenceSession(str(destination), providers=['CPUExecutionProvider']).run(
        None, {'image': image.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('spatial', [False, True])
def test_normalized_autoencoder_has_input_dependent_outputs_and_gradients(spatial):
    torch.manual_seed(42)
    model = Autoencoder(base=8, latent=64, normalized=True, spatial=spatial)
    images = torch.rand(2, 3, 128, 128)
    outputs = model(images)
    assert float((outputs[0] - outputs[1]).detach().abs().mean()) > 1e-4
    reconstruction(outputs, images).backward()
    assert sum(p.grad.abs().sum() for p in model.compress.parameters()) > 0
    assert model.encoder[0].weight.grad.abs().sum() > 0


def test_backup_preserves_live_database_and_task_checkpoint(tmp_path):
    import sqlite3
    from restoration.backup import snapshot

    source, destination = tmp_path / "artifacts", tmp_path / "persistent"
    checkpoint = source / "checkpoints/universal/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint fixture")
    with sqlite3.connect(source / "tracking.db") as database:
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("CREATE TABLE runs (name TEXT)")
        database.execute("INSERT INTO runs VALUES ('verified')")
        database.commit()
        snapshot(source, destination, "universal")
    assert (destination / "checkpoints/universal/best.pt").read_bytes() == checkpoint.read_bytes()
    with sqlite3.connect(destination / "tracking.db") as database:
        assert database.execute("SELECT name FROM runs").fetchone() == ("verified",)
    with pytest.raises(ValueError):
        snapshot(source, source / "nested")


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (128, 128), (160, 120, 80)).save(stream, format="PNG")
    return stream.getvalue()


def fixture_data(root):
    manifests = root / "manifests"
    rows = []
    rng = np.random.default_rng(42)
    for i in range(8):
        array = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        path = root / f"image_{i}.png"
        Image.fromarray(array).save(path)
        rows.append({"id": str(i), "image": path.name})
    write_json(manifests / "pets.json", {"train": rows, "validation": rows, "test": []})
    validation = [{**rows[i], "corruption": config(kind, 42 + i, "low" if kind != "clean" else None)} for i, kind in enumerate(CLASSES)]
    write_json(manifests / "pets_validation.json", validation)
    face_rows = [{**row, "target": row["image"], "style": i % 3} for i, row in enumerate(rows)]
    write_json(manifests / "fs2k.json", {"train": face_rows, "validation": face_rows, "test": []})


@pytest.mark.parametrize("kind", CLASSES)
def test_corruption_repeatability_and_range(kind):
    source = np.full((128, 128, 3), .5, dtype=np.float32)
    cfg = config(kind, 42, "high" if kind != "clean" else None)
    first = apply(source, cfg)
    np.testing.assert_array_equal(first, apply(source, cfg))
    assert first.dtype == np.float32 and first.min() >= 0 and first.max() <= 1
    if kind == "salt":
        selected = np.any(first != source, axis=2)
        assert .13 < selected.mean() < .17
        assert np.all((first[selected] == 0) | (first[selected] == 1))
    if kind == "occlusion":
        assert abs(np.all(first == 0, axis=2).mean() - .35) < .005


def test_balanced_batches_have_every_label():
    sampler = BalancedBatches(40, 8)
    batches = list(sampler)
    assert len(batches) == 5
    assert len({index for batch in batches for index, _ in batch}) == 40
    for batch in batches:
        assert np.bincount([label for _, label in batch]).tolist() == [2, 2, 2, 2]


def test_paired_augmentation_preserves_alignment(tmp_path):
    fixture_data(tmp_path)
    faces = Faces(tmp_path)
    for i in range(len(faces)):
        image, target, _ = faces[i]
        torch.testing.assert_close(image, target)


def test_ssim_identity_and_error():
    image = torch.rand(2, 3, 128, 128)
    torch.testing.assert_close(ssim(image, image), torch.ones(2), atol=1e-5, rtol=1e-5)
    assert torch.all(ssim(image, 1 - image) < .1)


def test_compressed_autoencoder_and_soft_gradient():
    image = torch.rand(2, 3, 128, 128)
    experts = [Autoencoder(base=8, latent=16) for _ in range(3)]
    assert experts[0].compress.out_features < image[0].numel()
    mixture = SoftMixture(Classifier(base=8), experts)
    output, weights, _ = mixture(image)
    assert output.shape == image.shape
    torch.testing.assert_close(weights.sum(1), torch.ones(2))
    assert torch.all(output >= 0) and torch.all(output <= 1)
    reconstruction(output, image).backward()
    assert mixture.gate.features[0].weight.grad.abs().sum() > 0
    assert all(expert.compress.weight.grad.abs().sum() > 0 for expert in experts)


def test_gan_style_condition_affects_both_networks():
    image = torch.rand(2, 3, 128, 128)
    generator, discriminator = Generator(base=8, embedding=4, dropout=0), Discriminator(base=8, embedding=4)
    generator.eval()
    zero, one = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long)
    first, second = generator(image, zero), generator(image, one)
    assert first.shape == image.shape and not torch.allclose(first, second)
    assert not torch.allclose(discriminator(image, first, zero), discriminator(image, first, one))
    discriminator(image, first, zero).mean().backward()
    assert generator.style.weight.grad.abs().sum() > 0
    assert discriminator.style.weight.grad.abs().sum() > 0


def test_api_missing_models_and_upload_validation(tmp_path, monkeypatch):
    from backend import app as api
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path)
    client = TestClient(api.app)
    assert client.get("/api/health").json()["status"] == "awaiting_models"
    upload = {"file": ("photo.png", image_bytes(), "image/png")}
    for endpoint in ("universal-restoration", "hard-routing", "soft-mixture", "face-to-sketch"):
        assert client.post(f"/api/{endpoint}", files=upload).status_code == 503
    assert client.post("/api/corrupt", files={"file": ("bad.png", b"bad", "image/png")}).status_code == 400
    assert client.post("/api/corrupt", files=upload, data={"corruption": "salt", "severity": "high"}).status_code == 200
    assert client.post("/api/face-to-sketch", files=upload, data={"style": 9}).status_code == 422
    assert client.post("/api/corrupt", files=upload, data={"seed": -1}).status_code == 422


def test_archive_rejects_traversal(tmp_path):
    import zipfile
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "extracted")


def test_training_resume_and_export(tmp_path, monkeypatch):
    """One real optimization epoch on synthetic data; verify resume and ONNX."""
    from restoration.train import DEFAULT, train_task, load_checkpoint
    from restoration.export import export_one
    root, artifacts = tmp_path / "data", tmp_path / "artifacts"
    root.mkdir(); artifacts.mkdir()
    fixture_data(root)
    cfg = {**DEFAULT, "base": 8, "latent": 16, "batch_size": 4, "embedding": 4}
    args = SimpleNamespace(data=root, artifacts=artifacts, device="cpu", workers=0,
                           max_train_batches=1, max_validation_batches=1, warmup_epochs=1, synthetic=True)
    exported = {}
    for task in ("universal", "classifier", "salt", "blur", "occlusion", "soft", "gan"):
        directory = artifacts / "checkpoints" / task
        score = train_task(task, cfg, args, directory, epochs=1)
        assert np.isfinite(score)
        checkpoint = directory / "best.pt"
        assert load_checkpoint(checkpoint)["synthetic"]
        with pytest.raises(ValueError, match="synthetic"):
            export_one(checkpoint, artifacts / "models")
        result = export_one(checkpoint, artifacts / "models", permit_synthetic=True)
        exported[task] = result
        assert result["verified"] and max(c["max_absolute_error"] for c in result["checks"]) < 1e-4
    # Resume an existing contiguous checkpoint with the alternative CPU layout.
    args.cpu_channels_last = True
    train_task("universal", cfg, args, artifacts / "checkpoints/universal", epochs=2)
    assert load_checkpoint(artifacts / "checkpoints/universal/last.pt")["epoch"] == 1
    assert len(json.loads((artifacts / "checkpoints/universal/history.json").read_text())) == 2
    assert load_checkpoint(artifacts / "checkpoints/universal/last.pt")['execution_memory_format'] == 'channels_last'
    exported['universal'] = export_one(artifacts/'checkpoints/universal/last.pt',artifacts/'models',permit_synthetic=True)
    assert exported['universal']['verified']
    # Test-only harness exposes synthetic graphs; the real API still rejects them.
    from backend import app as api
    monkeypatch.setattr(api, "registered_model", lambda name: (artifacts / "models" / f"{name}.onnx", exported[name]))
    api.verified_session.cache_clear()
    client = TestClient(api.app)
    upload = {"file": ("fixture.png", image_bytes(), "image/png")}
    for endpoint in ("universal-restoration", "hard-routing", "soft-mixture", "face-to-sketch"):
        response = client.post(f"/api/{endpoint}", files=upload, data={"corruption": "salt", "style": 2})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["output"].startswith("data:image/png;base64,") and result["inference_ms"] >= 0
        if endpoint == "hard-routing":
            assert abs(sum(result["probabilities"]) - 1) < 1e-5
        if endpoint == "soft-mixture":
            assert abs(sum(result["weights"]) - 1) < 1e-5
    api.verified_session.cache_clear()
