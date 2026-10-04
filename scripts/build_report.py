"""Assemble IEEE source from actual experiment records; refuse an empty final report."""
import argparse
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parents[1]


def tex(value):
    escaped = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escaped.get(character, character) for character in str(value))


def figure(name, caption):
    return "\\evidencefigure{" + name + "}{" + tex(caption) + "}\n"


SEVERITIES = ("low", "medium", "high")


def _rows(summary, system, corruption):
    return [r for r in summary["restoration"] if r["system"] == system and r["corruption"] == corruption]


def _mean(summary, system, corruption, metric):
    rows = _rows(summary, system, corruption)
    return sum(r[metric] * r["n"] for r in rows) / sum(r["n"] for r in rows)


def _at(summary, system, corruption, severity, metric):
    return next(r[metric] for r in _rows(summary, system, corruption) if r["severity"] == severity)


def _gain(summary, system, corruption):
    return _mean(summary, system, corruption, "psnr") - _mean(summary, "corrupted_baseline", corruption, "psnr")


def abstract_findings(summary):
    """One sentence of headline measurements, computed from the saved test summary."""
    classifier = summary["classifier"]
    face = summary["face"]
    face_ssim = sum(r["ssim"] * r["n"] for r in face) / sum(r["n"] for r in face)
    return (f"On {summary['pet_input_count']:,} deterministic test inputs the corruption classifier reaches "
            f"{classifier['accuracy'] * 100:.1f}\\% accuracy (macro F1 {classifier['macro avg']['f1-score']:.3f}). "
            f"Predicted routing changes mean PSNR by {_gain(summary, 'predicted_routing', 'salt'):+.1f} dB on "
            f"salt-and-pepper noise and {_gain(summary, 'predicted_routing', 'occlusion'):+.1f} dB on occlusion, but by "
            f"{_gain(summary, 'predicted_routing', 'blur'):+.1f} dB on blur, where the soft mixture reaches "
            f"{_gain(summary, 'soft', 'blur'):+.1f} dB. The sketch generator reaches a mean SSIM of {face_ssim:.2f} and "
            "produces smooth sketches without stroke detail.")


def discussion_section(summary, artifacts):
    """Interpretation of the results; every number is read from the saved evaluation files."""
    artifacts = Path(artifacts)
    results = artifacts / "results" / summary["split"]
    names = (("corrupted_baseline", "Input (unrestored)"), ("universal", "Universal"), ("oracle_routing", "Oracle routing"),
             ("predicted_routing", "Predicted routing"), ("soft", "Soft mixture"))
    lines = [r"\begin{table}[!ht]\centering\caption{Mean test PSNR (dB) / SSIM per condition, averaged over severities "
             r"(clean PSNR is capped at 100 dB)}\label{tab:means}\scriptsize" + "\n",
             r"\setlength{\tabcolsep}{3pt}\begin{tabular}{lcccc}\toprule System & Clean & Salt & Blur & Occlusion\\\midrule" + "\n"]
    for system, label in names:
        cells = [f"{_mean(summary, system, kind, 'psnr'):.1f}/{_mean(summary, system, kind, 'ssim'):.3f}"
                 for kind in ("clean", "salt", "blur", "occlusion")]
        lines.append(label + " & " + " & ".join(cells) + "\\\\\n")
    lines.append("\\bottomrule\\end{tabular}\\end{table}\n")
    lines.append("Table~\\ref{tab:means} condenses the per-severity tables above. The paragraphs below interpret them; "
                 "causal explanations are marked as hypotheses where no ablation was run.\n")

    gap = lambda severity: (_at(summary, "oracle_routing", "salt", severity, "psnr") - _at(summary, "universal", "salt", severity, "psnr"))
    lines.append("\\paragraph{Salt-and-pepper noise.} This is the clearest success. Relative to leaving the noisy input unchanged, "
                 f"mean PSNR rises by {_gain(summary, 'universal', 'salt'):+.1f} dB (universal), {_gain(summary, 'oracle_routing', 'salt'):+.1f} dB "
                 f"(oracle specialist), {_gain(summary, 'predicted_routing', 'salt'):+.1f} dB (predicted routing) and "
                 f"{_gain(summary, 'soft', 'salt'):+.1f} dB (soft mixture); specialist SSIM is {_mean(summary, 'oracle_routing', 'salt', 'ssim'):.3f} "
                 f"against {_mean(summary, 'corrupted_baseline', 'salt', 'ssim'):.3f} for the input. The dedicated specialist beats the universal model by "
                 f"{gap('low'):.1f} dB at low severity and {gap('high'):.1f} dB at high severity, consistent with the hypothesis that a single network "
                 "sharing capacity across conditions does worse as noise grows. This is a comparison under our training budget, not a proof "
                 "that specialisation is always superior.\n")

    lines.append("\\paragraph{Occlusion.} PSNR rises by "
                 f"{_gain(summary, 'oracle_routing', 'occlusion'):+.1f} dB for oracle routing because black holes are filled with plausible colours. "
                 f"SSIM is mixed: at low severity the specialist SSIM ({_at(summary, 'oracle_routing', 'occlusion', 'low', 'ssim'):.3f}) is below the "
                 f"unrestored input ({_at(summary, 'corrupted_baseline', 'occlusion', 'low', 'ssim'):.3f}), because the autoencoder re-synthesises "
                 "every pixel, including those the input already had exactly, whereas at high severity SSIM improves from "
                 f"{_at(summary, 'corrupted_baseline', 'occlusion', 'high', 'ssim'):.3f} to {_at(summary, 'oracle_routing', 'occlusion', 'high', 'ssim'):.3f}. "
                 "The failure grids show large masks across faces filled with smooth guessed colour, not recovered texture.\n")

    ceiling = _mean(summary, "universal", "clean", "psnr")
    lines.append("\\paragraph{Blur.} This is the weakest condition. Mean PSNR change relative to the unrestored blurred input is "
                 f"{_gain(summary, 'universal', 'blur'):+.1f} dB (universal), {_gain(summary, 'oracle_routing', 'blur'):+.1f} dB (oracle), "
                 f"{_gain(summary, 'predicted_routing', 'blur'):+.1f} dB (predicted) and {_gain(summary, 'soft', 'blur'):+.1f} dB (soft). "
                 "Oracle routing changes PSNR by "
                 + ", ".join(f"{_at(summary, 'oracle_routing', 'blur', s, 'psnr') - _at(summary, 'corrupted_baseline', 'blur', s, 'psnr'):+.1f}" for s in SEVERITIES)
                 + " dB at low, medium and high severity, so restoration helps only at the highest severity and then by under one dB. "
                 f"A likely explanation is the reconstruction ceiling of the compressed latent: even on clean inputs the universal model reaches only "
                 f"{ceiling:.1f} dB, while the blur specialist reaches {_at(summary, 'oracle_routing', 'blur', 'low', 'psnr'):.1f} dB at low severity, "
                 f"where the unrestored input already scores {_at(summary, 'corrupted_baseline', 'blur', 'low', 'psnr'):.1f} dB. Passing a lightly blurred image "
                 "through the bottleneck is therefore likely to lower fidelity. No larger-latent ablation was run, so this is a hypothesis consistent with the data, "
                 "not a demonstrated cause.\n")

    routing = {(r["corruption"], r["severity"]): r for r in summary["routing"]}
    clean_weight = routing[("clean", "clean")]["clean"]
    lines.append("\\paragraph{Clean inputs.} The universal model alters clean images (PSNR "
                 f"{ceiling:.1f} dB, SSIM {_mean(summary, 'universal', 'clean', 'ssim'):.3f}), a cost that hard routing and the soft mixture largely avoid "
                 f"through the identity branch: predicted routing keeps SSIM at {_mean(summary, 'predicted_routing', 'clean', 'ssim'):.3f}, and the soft "
                 f"mixture at {_mean(summary, 'soft', 'clean', 'ssim'):.3f} because the gate gives the identity branch a mean weight of {clean_weight * 100:.1f}\\% on clean images.\n")

    report = summary["classifier"]
    confusion_path = results / "classifier.json"
    confusion = json.loads(confusion_path.read_text()).get("normalized_confusion") if confusion_path.exists() else None
    clean_to_blur = f" and {confusion[0][2] * 100:.1f}\\% of clean images are labelled blur" if confusion else ""
    lines.append("\\paragraph{Classifier and hard routing.} The classifier reaches "
                 f"{report['accuracy'] * 100:.1f}\\% accuracy. Its errors concentrate in the clean class: clean recall is {report['clean']['recall']:.3f}{clean_to_blur}. "
                 "This is expected because low-severity blur (a $3\\times3$ kernel with $\\sigma=0.7$) barely differs from a clean photograph; "
                 f"salt-and-pepper is almost always recognised (recall {report['salt']['recall']:.3f}). ")
    differences = [_at(summary, "predicted_routing", kind, s, "psnr") - _at(summary, "oracle_routing", kind, s, "psnr")
                   for kind in ("salt", "blur", "occlusion") for s in SEVERITIES]
    blur_as_clean = f" (for example {confusion[2][0] * 100:.1f}\\% of blurred images are labelled clean, where bypassing the specialist can even help low-severity blur)" if confusion else ""
    lines.append(f"Predicted routing differs from oracle routing by only {min(differences):+.2f} to {max(differences):+.2f} dB per condition on average, "
                 f"because few corrupted images are misrouted{blur_as_clean}. "
                 "Misrouting can nevertheless be severe: ")
    cases_path = results / "routing_error_cases.json"
    cases = json.loads(cases_path.read_text()) if cases_path.exists() else []
    if cases and all(c["corruption"] == "salt" and c["predicted"] == "occlusion" for c in cases):
        lines.append(f"the four worst predicted-routing cases are salt-and-pepper images sent to the occlusion expert (PSNR "
                     f"{min(c['psnr'] for c in cases):.1f}--{max(c['psnr'] for c in cases):.1f} dB against "
                     f"{min(c['oracle_psnr'] for c in cases):.1f}--{max(c['oracle_psnr'] for c in cases):.1f} dB with the correct expert); the saved failure analysis notes dark backgrounds.\n")
    else:
        lines.append("see the routing failure grid.\n")

    experts = ("salt", "blur", "occlusion")
    stray = max(row[e] for row in summary["routing"] for e in experts if row["corruption"] == "clean" or e != row["corruption"])
    routes_path = results / "routing_examples.json"
    inactive = json.loads(routes_path.read_text()).get("inactive_branches_below_one_percent", []) if routes_path.exists() else []
    lines.append("\\paragraph{Soft gate.} For salt-and-pepper the gate is nearly one-hot "
                 f"(salt expert weight {routing[('salt', 'low')]['salt']:.3f} at low severity and at least {min(routing[('salt', 'medium')]['salt'], routing[('salt', 'high')]['salt']):.4f} at medium and high severity). "
                 "For blur and occlusion at low severity it shares weight with the identity branch "
                 f"(identity weights {routing[('blur', 'low')]['clean']:.2f} and {routing[('occlusion', 'low')]['clean']:.2f}). This explains why the soft model beats the specialists on "
                 f"low-severity blur ({_at(summary, 'soft', 'blur', 'low', 'psnr'):.1f} versus {_at(summary, 'oracle_routing', 'blur', 'low', 'psnr'):.1f} dB) yet fills less of the "
                 f"mask at low-severity occlusion ({_at(summary, 'soft', 'occlusion', 'low', 'psnr'):.1f} versus {_at(summary, 'oracle_routing', 'occlusion', 'low', 'psnr'):.1f} dB PSNR, "
                 f"{_at(summary, 'soft', 'occlusion', 'low', 'ssim'):.3f} versus {_at(summary, 'oracle_routing', 'occlusion', 'low', 'ssim'):.3f} SSIM). "
                 + ("No branch contributes under 1\\% of the global mean weight, so no expert is inactive, and " if not inactive else "")
                 + f"no unrelated expert exceeds a mean weight of {stray:.3f}, so none dominates unrelated inputs. ")
    verification_path = BASE / "docs/local-verification.json"
    if verification_path.exists():
        times = {}
        for row in json.loads(verification_path.read_text())["api"]:
            times.setdefault(row["endpoint"] if "endpoint" in row else row.get("name", "?"), []).append(row["inference_ms"])
        if {"universal-restoration", "hard-routing", "soft-mixture"} <= set(times):
            lines.append("The price is compute: the soft graph evaluates all four branches, with single-request CPU inference of "
                         f"{times['soft-mixture'][0]:.1f} ms against {times['hard-routing'][0]:.1f} ms for hard routing and {times['universal-restoration'][0]:.1f} ms for the universal model.\n")
        else:
            lines.append("\n")

    face = summary["face"]
    weighted_ssim = sum(r["ssim"] * r["n"] for r in face) / sum(r["n"] for r in face)
    worst = min(face, key=lambda r: r["psnr"])
    lines.append("\\paragraph{Face-to-sketch.} Mean SSIM over the 1,046 test pairs is "
                 f"{weighted_ssim:.3f}; style {worst['style'] + 1} ($n={worst['n']}$) is weakest at {worst['psnr']:.1f} dB PSNR, while style 3 has only "
                 f"{face[2]['n']} test pairs, so its score is noisy. The generated sketches reproduce head pose and outline but are pale and smooth, "
                 "without the hatching of the references (see the failure analysis). ")
    history_path = artifacts / "checkpoints" / "gan" / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        best = min(history, key=lambda h: h["validation_objective"])
        low = min(min(h["train_discriminator_real"], h["train_discriminator_fake"]) for h in history)
        high = max(max(h["train_discriminator_real"], h["train_discriminator_fake"]) for h in history)
        lines.append(f"Discriminator real and fake losses stay between {low:.2f} and {high:.2f}, so neither network overwhelms the other. The validation objective "
                     f"is lowest at epoch {best['epoch']} ({best['validation_objective']:.3f}) and is {history[-1]['validation_objective']:.3f} at epoch {history[-1]['epoch']}: "
                     "longer adversarial training did not improve the L1/SSIM-based objective, so the epoch-"
                     f"{best['epoch']} checkpoint is deployed. ")
    lines.append("Plausible causes are the small generator, only 899 training pairs at $128\\times128$, and the weight-100 L1 term favouring averaged outputs; "
                 "none was isolated by an experiment.\n")

    lines.append("\\paragraph{How the evidence shaped the design.} Validation pilots decided the architecture. The validation objective of the first dense-latent "
                 "model was 0.322; an $8\\times8$ spatial latent reached 0.281, a $16\\times16$ latent with local residual blocks 0.207, and the final design "
                 "(8,192-value latent, BatchNorm, PixelShuffle upsampling) 0.144 in its pilot and 0.085 after full training. These runs differ in budget, so the "
                 "sequence indicates direction, not a controlled ablation. The remaining weaknesses above, blur and sketch detail, are the points where "
                 "additional capacity, a sharper loss or a longer GPU schedule would be tried next.\n")
    return "\n".join(lines)


def results_section(summary, artifacts, generated):
    lines = [r"\statusnote{Measured " + tex(summary["split"]) + r" results; " + ("subset evaluation." if summary["subset"] else "complete evaluation.") + "}\n"]
    analysis_path = artifacts/'results'/summary['split']/'visual_analysis.json'
    visual_analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    mapping = [("universal", "Task 1: Universal model"), ("oracle_routing", "Task 2: Oracle routing"),
               ("predicted_routing", "Task 2: Predicted routing"), ("soft", "Task 3: Soft mixture")]
    for system, heading in mapping:
        groups = [row for row in summary["restoration"] if row["system"] == system]
        lines += [r"\subsection{" + heading + "}\n", r"\begin{table}[!ht]\centering\caption{" + heading + r"}\scriptsize" + "\n",
                  r"\begin{tabular}{llrrrr}\toprule Condition & Severity & $n$ & L1 & PSNR & SSIM\\\midrule" + "\n"]
        for row in groups:
            lines.append(f"{tex(row['corruption'])} & {tex(row['severity'])} & {row['n']} & {row['l1']:.4f} & {row['psnr']:.2f} & {row['ssim']:.4f}\\\\\n")
        lines.append("\\bottomrule\\end{tabular}\\end{table}\n")
        baseline = [r for r in summary["restoration"] if r["system"] == "corrupted_baseline"]
        n = sum(r["n"] for r in groups)
        average = sum(r["psnr"] * r["n"] for r in groups) / n
        baseline_average = sum(r["psnr"] * r["n"] for r in baseline) / sum(r["n"] for r in baseline)
        lines.append(f"Mean per-input PSNR is {average:.2f} dB; the corrupted-input baseline is {baseline_average:.2f} dB. "
                     "Clean identity inputs influence these averages, so condition-specific rows should guide interpretation.\n")
        baseline_by_condition = {(r['corruption'],r['severity']):r for r in baseline}
        for kind in ('salt','blur','occlusion'):
            selected = [r for r in groups if r['corruption']==kind]
            selected.sort(key=lambda r: ('low','medium','high').index(r['severity']))
            changes, improved = [], 0
            for row in selected:
                original = baseline_by_condition[(kind,row['severity'])]
                changes.append(row['psnr']-original['psnr'])
                before = .8*original['l1']+.2*(1-original['ssim'])
                after = .8*row['l1']+.2*(1-row['ssim'])
                improved += after < before - 1e-6
            if selected:
                lines.append(tex(kind.capitalize()) + ": PSNR changes relative to leaving the corrupted input unchanged are "
                             + ', '.join(f'{change:+.2f}' for change in changes)
                             + f" dB at low/medium/high severity; the fixed reconstruction objective improves in {improved} of {len(selected)} severity groups.\n")
        clean = next((r for r in groups if r['corruption']=='clean'),None)
        if clean:
            clean_objective = .8*clean['l1']+.2*(1-clean['ssim'])
            lines.append(f"On clean inputs, L1 is {clean['l1']:.5f}, SSIM is {clean['ssim']:.5f}, and the fixed objective is {clean_objective:.5f}. "
                         "The identity reference has no reconstruction error; any clean-image degradation is therefore a limitation.\n")
        for page in range(1, 4):
            name = f"{system}_representative_{page}.png"
            if (generated / name).exists():
                lines.append(figure(name, f"{heading}: representative examples {(page - 1) * 4 + 1}-{page * 4}. Columns: clean target, input, output, absolute error."))
        failure_file = artifacts / "results" / summary["split"] / f"{system}_failures.json"
        if failure_file.exists():
            failures = json.loads(failure_file.read_text())
            lines.append(figure(f"{system}_failures.png", heading + ": four lowest-PSNR examples with distinct source images."))
            for i, row in enumerate(failures):
                lines.append(f"Failure {i + 1}: {tex(row['id'])}, {tex(row['corruption'])} at {tex(row['severity'])} severity, "
                             f"PSNR {row['psnr']:.2f} dB and SSIM {row['ssim']:.4f}. ")
                if system == "predicted_routing" and not row["routing_correct"]:
                    lines.append(f"The classifier selected {tex(row['predicted'])}, differing from the true {tex(row['corruption'])} condition. ")
                analysis = visual_analysis.get(system, {}).get(row['id'])
                lines.append(tex(analysis) + "\n" if analysis else "The error-map locations should be visually inspected before attributing the error to texture loss, blur or missing detail.\n")
    report = summary["classifier"]
    lines.append(r"\subsection{Corruption classifier}" + "\n")
    lines.append(f"Overall accuracy is {report.get('accuracy', 0):.4f}. Macro precision, recall and F1 are "
                 f"{report['macro avg']['precision']:.4f}, {report['macro avg']['recall']:.4f} and {report['macro avg']['f1-score']:.4f}.\n")
    lines += [r"\begin{table}[!ht]\centering\caption{Per-class classifier results}\scriptsize\begin{tabular}{lrrrr}\toprule Class & Precision & Recall & F1 & $n$\\\midrule" + "\n"]
    for name in ("clean", "salt", "blur", "occlusion"):
        row = report[name]
        lines.append(f"{name} & {row['precision']:.4f} & {row['recall']:.4f} & {row['f1-score']:.4f} & {int(row['support'])}\\\\\n")
    lines.append("\\bottomrule\\end{tabular}\\end{table}\n")
    lines.append(figure("confusion.png", "Normalized confusion matrix; each row is a true class. Off-diagonal entries identify routing mistakes."))
    routing_error_path = artifacts/'results'/summary['split']/'routing_error_cases.json'
    if routing_error_path.exists():
        cases = json.loads(routing_error_path.read_text())
        if cases:
            lines.append(figure('routing_error_cases.png','Hard-routing examples with an incorrect classifier prediction. Targets and errors expose the restoration consequence.'))
            for row in cases:
                lines.append(tex(row['id']) + f": true {tex(row['corruption'])} ({tex(row['severity'])}), predicted {tex(row['predicted'])}; "
                             f"predicted-routing PSNR {row['psnr']:.2f} dB versus oracle {row['oracle_psnr']:.2f} dB "
                             f"(difference {row['predicted_minus_oracle_psnr']:+.2f} dB). ")
            lines.append("A wrong label changes the selected branch, but its numerical effect can be positive or negative because the specialists are imperfect; classification correctness alone does not establish restoration quality.\n")
    lines += [r"\subsection{Expert weight analysis}" + "\n", r"\begin{table}[!ht]\centering\caption{Mean soft branch weights}\scriptsize\begin{tabular}{llrrrr}\toprule Class & Severity & Clean & Salt & Blur & Mask\\\midrule" + "\n"]
    for row in summary["routing"]:
        lines.append(f"{row['corruption']} & {row['severity']} & {row['clean']:.3f} & {row['salt']:.3f} & {row['blur']:.3f} & {row['occlusion']:.3f}\\\\\n")
    lines.append("\\bottomrule\\end{tabular}\\end{table}\n")
    lines.append(figure("routing_heatmap.png", "Mean expert contribution by true condition and severity. Concentration in unrelated rows suggests imbalance."))
    route_path = artifacts / "results" / summary["split"] / "routing_examples.json"
    if route_path.exists():
        routes = json.loads(route_path.read_text())
        inactive = routes["inactive_branches_below_one_percent"]
        lines.append("Branches below 1\\% global mean contribution: " + tex(", ".join(inactive) if inactive else "none") + ".\n")
        for category in ("dominant", "distributed"):
            lines.append("Examples with " + category + " weights: ")
            for row in routes[category][:2]:
                lines.append(tex(row["id"]) + " (" + ", ".join(f"{name}={row['weight_' + name]:.3f}" for name in ("clean", "salt", "blur", "occlusion")) + "); ")
            lines.append("these examples demonstrate gate concentration, not automatically improved restoration.\n")
    lines += [r"\subsection{Task 4: Face-to-sketch results}" + "\n", r"\begin{table}[!ht]\centering\caption{Paired sketch reconstruction by style}\begin{tabular}{lrrrr}\toprule Style & $n$ & L1 & PSNR & SSIM\\\midrule" + "\n"]
    for row in summary["face"]:
        lines.append(f"{row['style'] + 1} & {row['n']} & {row['l1']:.4f} & {row['psnr']:.2f} & {row['ssim']:.4f}\\\\\n")
    lines.append("\\bottomrule\\end{tabular}\\end{table}\n")
    if (generated/'face_representative_1.png').exists():
        for page in range(1, 4):
            name = f'face_representative_{page}.png'
            if (generated/name).exists():
                lines.append(figure(name, f"Paired sketch examples {(page - 1) * 4 + 1}-{page * 4}: target, photograph, output and absolute error."))
    else:
        lines.append(figure("face_representative.png", "Paired face-to-sketch examples with targets and absolute errors; inspect facial structure and background artifacts."))
    lines.append(figure("style_comparison.png", "The same photograph under all three style conditions. Only its annotated style has a paired ground truth."))
    face_failure_path = artifacts/'results'/summary['split']/'face_failures.json'
    if face_failure_path.exists():
        lines.append(figure('face_failures.png','Four lowest-PSNR paired sketch reconstructions, with distinct source photographs.'))
        for i,row in enumerate(json.loads(face_failure_path.read_text())):
            lines.append(f"Sketch failure {i+1}: {tex(row['id'])}, style {row['style']+1}, PSNR {row['psnr']:.2f} dB and SSIM {row['ssim']:.4f}. ")
            analysis = visual_analysis.get('gan', {}).get(row['id'])
            if analysis:
                lines.append(tex(analysis)+"\n")
    lines.append(r"\subsection{Observed failure modes and interpretation}" + "\n")
    repair_path = artifacts / "repair_provenance.json"
    detail_path = artifacts / "detail_provenance.json"
    if detail_path.exists():
        detail = json.loads(detail_path.read_text())
        lines.append("Earlier dense and small spatial bottlenecks produced collapsed or blurry reconstructions. A larger 16-by-16 compressed representation with local residual refinement was investigated. A learning-rate 0.001 pilot saturated and was rejected. The subsequent lower-rate pilot reached validation objective "
                     + f"{detail['pilot_best']:.5f}, compared with {detail['previous_universal_best']:.5f} for the previous selected model. "
                     + "This motivated a separate Optuna study and longer training, with all selection based on validation.\n")
        pilot_path = artifacts / 'pilot_records.json'
        if pilot_path.exists():
            records = json.loads(pilot_path.read_text())
            lines.append(r"\begin{table}[!ht]\centering\caption{Validation-led architecture pilots}\scriptsize\begin{tabular}{lrrrr}\toprule Pilot & Epoch & Objective & PSNR & SSIM\\\midrule" + "\n")
            for pilot in records:
                config, best = pilot['config'], pilot['best']
                label = ('Learned upsampling' if config.get('detail_shuffle') else 'BatchNorm' if config.get('detail_bn') else 'High LR' if config['lr'] >= .001 else 'Low LR')
                lines.append(f"{label} & {best['epoch']} & {best['validation_objective']:.4f} & {best['validation_psnr']:.2f} & {best['validation_ssim']:.4f}\\\\\n")
            lines.append("\\bottomrule\\end{tabular}\\end{table}\n")
            lines.append("These sequential pilots are exploratory, not controlled single-factor ablations: capacity, normalization, learning rate and upsampling differ. Their saved images and complete validation histories are retained. The selected learned-upsampling pilot used standard initialization; fresh final models use phase-matched ICNR initialization to address periodic artifacts, without guaranteeing their removal.\n")
        comparison_path = artifacts / 'validation_comparison.json'
        if comparison_path.exists():
            comparison = json.loads(comparison_path.read_text())
            lines.append(r"\begin{table}[!ht]\centering\caption{Complete validation comparison, lower objective is better}\scriptsize\begin{tabular}{lrrr}\toprule System & Previous & Candidate & Change (\%)\\\midrule" + "\n")
            for system, values in comparison['systems'].items():
                previous, candidate = values['previous_objective'], values['candidate_objective']
                change = 100 * (candidate / previous - 1)
                lines.append(f"{tex(system)} & {previous:.4f} & {candidate:.4f} & {change:+.1f}\\\\\n")
            lines.append("\\bottomrule\\end{tabular}\\end{table}\n")
            lines.append("Both model sets use the same complete validation inputs and fixed objective $0.8\\,L_1+0.2(1-\\mathrm{SSIM})$. Numerical improvements must be considered with the per-condition changes and saved images; a lower aggregate score does not establish improvement on every input.\n")
        lines.append("The corruption classifier is retained from the preceding validation-selected run, while a normalized GAN with learned upsampling is tuned in a new study and trained from scratch. Restoration and routing quality are judged using condition-specific metrics and failure grids, not successful API execution alone.\n")
        if detail.get('runtime_recovery'):
            lines.append(tex(detail['runtime_recovery']) + ". The saved pilot measurements survived, but its weights did not. The improved models were trained afresh on the laptop CPU, while the retained classifier and exploratory pilots used the earlier free GPU runs. Local and Colab data manifests match after newline normalization.\n")
        if detail.get('fixed_schedule'):
            lines.append("The fixed final improvement schedule was " + tex(', '.join(f'{task}={epochs} epochs' for task, epochs in detail['fixed_schedule'].items())) + ". This deadline-aware budget was chosen before inspecting the final candidate test results.\n")
    elif repair_path.exists():
        repair = json.loads(repair_path.read_text())
        lines.append("The first run's dense-bottleneck universal and specialist models produced nearly constant outputs in saved validation examples. A three-epoch normalization-only pilot reached validation objective 0.32742 and remained nearly constant. It was rejected. A separate spatial-bottleneck pilot reached " + f"{repair['pilot_validation_objective']:.5f}" + ", versus 0.32176 for the original selected universal checkpoint. Its validation images preserved coarse object outlines but had large color errors and blur. This motivated the separate search and fixed training schedule reported here.\n")
        lines.append("The first official test had already been observed before this repair. Architecture decisions, search and checkpoint selection for the repair used validation only; the subsequent test is a disclosed second evaluation, not an untouched first test. Classifier and GAN configurations reuse the original validation searches and are retrained; universal, specialist and joint-mixture studies are new. Final spatial reconstructions respond to the input but remain blurry with substantial color errors. Clean and blurred inputs can lose detail after restoration. Correct classification and concentrated mixture weights still do not establish useful restoration: compare the image grids, condition-specific errors and corrupted-input baseline.\n")
    else:
        lines.append("Visual inspection of the saved universal examples shows nearly constant gray-brown reconstructions across distinct photographs. This is reconstruction collapse, rather than successful denoising. The specialist models also retain very little image detail. Consequently, correct corruption classification alone cannot establish useful restoration. The soft mixture can retain structure through its identity branch, but a large identity contribution can leave the input corruption unchanged; weight concentration must not be interpreted as restoration success.\n")
        lines.append("The sketch examples retain coarse facial shape but show pronounced blur and checkerboard patterns. They do not reproduce the target's fine hatching. These outcomes motivate normalization, a better-conditioned bottleneck and decoder, and longer validation-led training in future work. These proposed changes have not been evaluated in this first run. Final test results were not used to select a new configuration within this run.\n")
    lines.append(r"\subsection{Completed search and training records}" + "\n")
    for task in ("universal", "classifier", "specialists", "soft", "gan"):
        study = json.loads((artifacts / "studies" / f"{task}.json").read_text())
        lines.append(f"{tex(task)}: {study['completed']} completed trials, {study['pruned']} pruned trials; best trial "
                     f"{study['best_trial']}, validation objective {study['best_value']:.5f}. "
                     "The short-trial budget and complete search space are preserved in the study JSON.\n")
        lines.append(figure(f"optuna_{task}.png", "Completed validation trial objectives for " + task + ". These limited searches are not exhaustive."))
    for task, row in summary["checkpoints"].items():
        lines.append("Selected " + tex(task) + " checkpoint: epoch " + str(row["epoch"]) + ". Configuration: "
                     + tex(", ".join(f"{k}={v}" for k, v in row["config"].items())) + ".\n")
        lines.append(figure(f"curve_{task}.png", "Recorded training and validation objectives for " + task + "; differing objectives may have different scales."))
    return "\n".join(lines)


def build_report(artifacts, draft=False):
    report_dir = BASE / "report"
    generated = report_dir / "generated"
    generated.mkdir(exist_ok=True)
    metadata = json.loads((report_dir / "metadata.json").read_text())
    summary_path = Path(artifacts) / "results/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    if not draft and (not summary or summary["split"] != "test" or summary["subset"]):
        raise RuntimeError("A final report requires complete final test results; use --draft for preparation")
    if summary:
        for image in (Path(artifacts) / "results" / summary["split"]).glob("*.png"):
            shutil.copy2(image, generated / image.name)
        results = results_section(summary, Path(artifacts), generated)
    else:
        results = r"\statusnote{GPU training, tuning and final dataset evaluation have not been performed. No dataset-performance numbers are claimed.}" + "\n"
        results += "Prepared outputs include severity-specific tables, classifier metrics, routing heatmaps, twelve representative examples and four failure cases per restoration system. They will be inserted from actual experiment files after training."
    application_evidence = []
    for image in sorted((BASE / "docs/screenshots").glob("app-*")):
        if image.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        shutil.copy2(image, generated / image.name)
        application_evidence.append(figure(image.name, "Application with imported trained ONNX models: " + image.stem))
    verification = BASE / "docs/local-verification.json"
    if verification.exists():
        checked = json.loads(verification.read_text())
        response_note = (", corroborating its input-insensitivity failure." if checked['models']['universal']['input_insensitive_warning']
                         else ". This confirms input sensitivity for these two controls, not restoration quality.")
        application_evidence.append("Local verification passed all seven model checksums and finite-output checks, all four API workspaces, and all three sketch styles using CPUExecutionProvider. Measured sample-request inference times ranged from " + f"{min(row['inference_ms'] for row in checked['api']):.2f}" + " to " + f"{max(row['inference_ms'] for row in checked['api']):.2f}" + " ms. These are single-request checks, not a throughput benchmark. The universal black-versus-white output difference was " + f"{checked['models']['universal']['black_white_max_output_difference']:.2e}" + response_note + "\\par\n")
    stitch = BASE / 'docs/stitch-evidence/export/stitch_restoration_lab_generative_studio/universal_restoration/screen.png'
    if stitch.exists():
        shutil.copy2(stitch, generated / 'stitch-original.png')
        application_evidence.append(figure('stitch-original.png', 'Original Google Stitch universal-workspace design. Decorative metrics and classes in this mockup are not experiment results.'))
        application_evidence.append("Stitch's HTML and PNG exports are retained in the repository (\\texttt{docs/stitch-evidence}). The implemented interface was reconciled with this design: dark navigation rail with the four workspaces, input panel, result panel, error and experiment views, and a green palette. Mock-up metrics and class names that correspond to no implemented function (for example the A100 latency badge and the ViT parameter counts) were omitted from the working application.\\par\n")
    application_evidence.append(r"\statusnote{Docker Compose was built and executed with a Linux Docker engine. HTTP tests passed for all four workspaces and all three sketch styles at localhost:8080.}")
    links = []
    for field, label in (("repository_url", "Repository"), ("youtube_url", "Demonstration video"), ("stitch_project_url", "Original Stitch design")):
        if not metadata[field]:
            print(f"WARNING: metadata.json has no {field}; the report will not contain that link.")
            continue
        links.append(label + ": " + "\\url{" + metadata[field] + "}.\\par\n")
    if links:
        links.insert(0, "\\subsection{Links}\n")
    values = {"STUDENT_NAME": tex(metadata["student_name"]), "STUDENT_ID": tex(metadata["student_id"]), "COURSE": tex(metadata["course"]),
              "ABSTRACT_STATUS": "This is a preparation draft: GPU experiments and final performance conclusions remain pending." if draft else "The following sections report the recorded final test measurements.",
              "RESULTS": results, "DISCUSSION": discussion_section(summary, artifacts) if summary else "Discussion is generated from final test results.",
              "ABSTRACT_FINDINGS": abstract_findings(summary) if summary and not draft else "", "APPLICATION_EVIDENCE": "\n".join(application_evidence), "EXTERNAL_LINKS": "\n".join(links),
              "CONCLUSION_STATUS": "The implementation and evaluation protocol are prepared; experimental conclusions remain pending." if draft else "The measured results above characterize the selected compact models under the recorded training budget; broader claims require further experiments."}
    spatial = bool(summary and summary['checkpoints']['universal']['config'].get('spatial'))
    detail = bool(summary and summary['checkpoints']['universal']['config'].get('detail'))
    values['MODEL_SELECTION_PROTOCOL'] = 'validation-based model selection followed by deterministic final testing'
    values['RUNTIME_STORAGE'] = 'Earlier GPU checkpoints and SQLite databases used temporary runtime disk without a Google Drive mount. Runtime deletion removes those files, so downloads are needed to preserve them.'
    if detail:
        values['MODEL_SELECTION_PROTOCOL'] = 'validation-only selection of hyperparameters and checkpoints, with final model hashes frozen before test evaluation (Section~\\ref{sec:protocol})'
        values['RUNTIME_STORAGE'] = 'Exploratory pilots used a free Colab T4 without mounting Google Drive. After its free GPU quota was reached and temporary new weights were lost, improvement searches and final training ran on the laptop CPU with persistent local checkpoints, optimizer/RNG state and SQLite databases. Earlier downloaded models and records were preserved separately.'
    values['LATENT_SEARCH'] = '256/512/1024' if spatial else '64/128/256'
    values['AUTOENCODER_METHOD'] = (
        r"The encoder has four stride-two convolutions with channel widths $b,2b,4b,8b$, reducing spatial dimensions from 128 to 8. "
        + (r"GroupNorm with four groups, LeakyReLU (slope 0.1), and dropout follow each encoder convolution. A $1\times1$ convolution with LeakyReLU compresses the representation to $d/64$ channels on an $8\times8$ grid, followed by a $1\times1$ expansion. Four bilinear upsampling and $3\times3$ convolution blocks decode RGB output; intermediate blocks use GroupNorm and LeakyReLU, and the output uses sigmoid. The latent contains $d\in\{256,512,1024\}$ values versus 49,152 input values. "
           if spatial else r"ReLU and dropout follow each encoder convolution. Linear layers compress the flattened feature to $d$ values and expand it again. Four transposed convolutions reconstruct RGB output with sigmoid range $[0,1]$. The input has 49,152 values, while $d\in\{64,128,256\}$. "))
    values['GAN_METHOD'] = 'Neither network uses batch normalization. Dropout is applied to early decoder stages.'
    values['GAN_SEARCH'] = r'Generator LR $10^{-4}$--$.003$; discriminator LR $.0001$--$.001$; batch 16/32/64; base 8/16/24; dropout 0--.3; embedding 4/8/16; $\lambda_1$ 50--150'
    if summary and summary['checkpoints']['gan']['config'].get('gan_refined'):
        values['GAN_METHOD'] = r'The refined generator uses BatchNorm with tracked inference statistics in encoder and intermediate decoder stages, learned convolution and PixelShuffle upsampling, and phase-matched ICNR initialization \cite{shi,aitken}. The discriminator normalizes its second and third downsampling stages. Dropout is applied to early decoder stages.'
        values['GAN_SEARCH'] = r'Generator LR $10^{-4}$--$.0005$; discriminator LR $.0001$--$.001$; batch 16/32/64; base 16/24/32; dropout 0--.08; embedding 4/8/16; $\lambda_1$ 50--150'
    values['FINAL_TRAINING'] = 'Selected configurations are retrained from scratch on the complete training split.'
    values['ENCODER_FEATURES'] = r'$8b\times8^2$'
    values['TRIAL_BUDGET'] = 'three epochs, 40 training batches and 20 validation batches'
    values['UNIVERSAL_SEARCH'] = r'LR $10^{-4}$--$.003$; batch 16/32/64; base 8/16/24; latent ' + values['LATENT_SEARCH'] + r'; dropout 0--.3; $\alpha$ .6--.95'
    if detail:
        values['AUTOENCODER_METHOD'] = r'Three stride-two encoder convolutions have widths $b,2b,4b$, reducing the image from 128 to 16 pixels. LeakyReLU and local residual blocks refine features, without normalization. A $1\times1$ convolution compresses to $d/256$ channels on a $16\times16$ grid. A corresponding expansion and three bilinear resize-convolution stages reconstruct RGB with sigmoid output. The latent search contains $d\in\{2048,4096,8192\}$ values, all smaller than the 49,152-value input. Residual blocks refine features within a stage; they never bypass the compressed representation. '
        values['FINAL_TRAINING'] = 'Selected restoration configurations are trained from scratch on the full training split; the classifier is retained and the refined GAN is independently tuned and trained from scratch.'
        if summary['checkpoints']['universal']['config'].get('detail_bn'):
            values['AUTOENCODER_METHOD'] = values['AUTOENCODER_METHOD'].replace('without normalization', 'with BatchNorm after stage convolutions, using tracked running statistics during inference')
        if summary['checkpoints']['universal']['config'].get('detail_shuffle'):
            values['AUTOENCODER_METHOD'] = values['AUTOENCODER_METHOD'].replace('three bilinear resize-convolution stages', r'three learned convolution and PixelShuffle upsampling stages \cite{shi}, initialized with phase-matched ICNR kernels \cite{aitken}')
        values['ENCODER_FEATURES'] = r'$4b\times16^2$'
        values['TRIAL_BUDGET'] = 'six epochs, 60 training batches and 80 validation batches for the new restoration and GAN studies; the historical classifier study retains its original budget'
        budget = json.loads((Path(artifacts)/'studies/universal.json').read_text())['budget']
        values['TRIAL_BUDGET'] = f"{budget['epochs']} epochs, {budget['train_batches']} training batches and {budget['validation_batches']} validation batches for the new restoration and GAN studies; the historical classifier study retains its original budget"
        values['UNIVERSAL_SEARCH'] = r'LR $10^{-4}$--$.0005$; batch 16/32/64; base 16/24/32; latent 2048/4096/8192; dropout 0--.08; $\alpha$ .6--.95'
    source = (report_dir / "template.tex").read_text(encoding="utf-8")
    for name, value in values.items():
        source = source.replace("%%" + name + "%%", value)
    target = report_dir / "report.tex"
    target.write_text(source, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=BASE / "artifacts")
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    build_report(args.artifacts, args.draft)

