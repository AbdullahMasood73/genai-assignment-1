# Google Stitch design prompt

Status: the original screens were generated and exported from
https://stitch.withgoogle.com/projects/16111510614845497849 . HTML and PNG exports
are preserved under `docs/stitch-evidence/export/`. The React implementation follows the design's
forest/ivory palette, dark sidebar with the four workspaces, input and result cards, and routing bars.
The mock-up's unsupported corruption names, metrics and style labels were omitted; the functional app
uses the assignment's actual classes and measured results. Original Stitch screenshots are distinct from
application screenshots.

Design a responsive web application named Restoration Lab for a university
generative imaging project. Use an elegant deep forest green sidebar, warm ivory
background, sage accents, thin borders, and understated scientific typography.

Four sidebar workspaces:
- Universal Restoration
- Hard-Routed Restoration
- Soft Mixture-of-Experts Restoration
- Face-to-Sketch Generator

Main screen: heading and short explanation; left controls card with drag-and-drop
image upload, input corruption dropdown, low/medium/high severity segmented
buttons, seed field and Restore image button; right card with input and restored
image panels side by side, inference time, download action, optional clean target
and absolute error map. Hard routing shows four classifier probabilities and
selected specialist. Soft mixture shows four expert weight bars. Sketch workspace
has webcam capture and three style choices. Include experiment records view,
clear missing-model and error states, and an accessible responsive mobile layout.
Create all four screens and a mobile variant.
