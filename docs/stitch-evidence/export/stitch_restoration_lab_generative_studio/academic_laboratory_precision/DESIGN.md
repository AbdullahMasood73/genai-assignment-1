---
name: Academic Laboratory Precision
colors:
  surface: '#f2fcf4'
  surface-dim: '#d2ddd5'
  surface-bright: '#f2fcf4'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#ecf6ee'
  surface-container: '#e6f0e8'
  surface-container-high: '#e1ebe3'
  surface-container-highest: '#dbe5dd'
  on-surface: '#151d19'
  on-surface-variant: '#424843'
  inverse-surface: '#29322d'
  inverse-on-surface: '#e9f3eb'
  outline: '#727973'
  outline-variant: '#c1c8c1'
  surface-tint: '#436651'
  primary: '#032717'
  on-primary: '#ffffff'
  primary-container: '#1b3d2b'
  on-primary-container: '#83a890'
  inverse-primary: '#a9cfb6'
  secondary: '#40674c'
  on-secondary: '#ffffff'
  secondary-container: '#bfeac8'
  on-secondary-container: '#446b50'
  tertiary: '#3f1300'
  on-tertiary: '#ffffff'
  tertiary-container: '#612301'
  on-tertiary-container: '#e4885e'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#c5ecd2'
  primary-fixed-dim: '#a9cfb6'
  on-primary-fixed: '#002112'
  on-primary-fixed-variant: '#2c4e3b'
  secondary-fixed: '#c2edcb'
  secondary-fixed-dim: '#a6d1b0'
  on-secondary-fixed: '#00210e'
  on-secondary-fixed-variant: '#294e35'
  tertiary-fixed: '#ffdbcd'
  tertiary-fixed-dim: '#ffb595'
  on-tertiary-fixed: '#360f00'
  on-tertiary-fixed-variant: '#76330f'
  background: '#f2fcf4'
  on-background: '#151d19'
  surface-variant: '#dbe5dd'
typography:
  display-lg:
    fontFamily: Newsreader
    fontSize: 2.75rem
    fontWeight: '500'
    lineHeight: 3.25rem
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Newsreader
    fontSize: 2rem
    fontWeight: '500'
    lineHeight: 2.5rem
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: Newsreader
    fontSize: 2rem
    fontWeight: '500'
    lineHeight: 2.5rem
    letterSpacing: -0.015em
  headline-lg-mobile:
    fontFamily: Newsreader
    fontSize: 1.5rem
    fontWeight: '500'
    lineHeight: 2rem
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Newsreader
    fontSize: 1.375rem
    fontWeight: '600'
    lineHeight: 1.875rem
  title-md:
    fontFamily: Inter
    fontSize: 1.0625rem
    fontWeight: '600'
    lineHeight: 1.5rem
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 1rem
    fontWeight: '400'
    lineHeight: 1.625rem
  body-md:
    fontFamily: Inter
    fontSize: 0.875rem
    fontWeight: '400'
    lineHeight: 1.375rem
  body-sm:
    fontFamily: Inter
    fontSize: 0.8125rem
    fontWeight: '400'
    lineHeight: 1.25rem
  metric-lg:
    fontFamily: JetBrains Mono
    fontSize: 1.25rem
    fontWeight: '500'
    lineHeight: 1.5rem
    letterSpacing: -0.02em
  metric-md:
    fontFamily: JetBrains Mono
    fontSize: 0.875rem
    fontWeight: '500'
    lineHeight: 1.25rem
  metric-sm:
    fontFamily: JetBrains Mono
    fontSize: 0.75rem
    fontWeight: '400'
    lineHeight: 1rem
  label-caps:
    fontFamily: Inter
    fontSize: 0.6875rem
    fontWeight: '600'
    lineHeight: 0.875rem
    letterSpacing: 0.06em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1.25rem
  gutter-desktop: 1.5rem
  margin: 1rem
  margin-tablet: 1.5rem
  margin-desktop: 2.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.25rem
---

## Brand & Style

This design system is engineered for advanced computational photography, generative image reconstruction, and archival restoration research. The aesthetic fuses archival scholarship with modern instrument precision—a digital laboratory monograph. It balances the dignity of university research institutions with the tactile efficiency of high-performance scientific tooling.

The interface evokes focus, technical authority, and calm discernment. Avoid neon accents, overt consumer tech glossy gradients, or hyper-stylized novelty. Instead, lean into high-density structural grids, measured contrast, editorial serif display framing, and legible monospaced data readouts that emphasize data integrity and peer-reviewed rigor.

## Colors

The palette grounds computationally intense operations in natural, archival tones:

- **Primary Brand Tier**: `#1B3D2B` (Core Oxford Forest Green), paired with `#0F261A` for deep chrome headers, interactive press states, and primary button fills. `#153324` functions as the dominant brand accent for primary highlights and selected states.
- **Canvas & Surface Tier**: The foundational backdrop is warm archival ivory (`#F9F8F3`), stepping down to `#F4F2EB` and `#ECE8DD` for recessed panels, parameter shelves, and segmented control backgrounds. Card surfaces sit on clean laboratory off-white (`#FFFFFF` or `#FCFBF9`).
- **Structural Outlines**: 1px subtle borders use muted sage-tinted neutrals (`#D8E0D8` and `#C9D5C9`) to define bounding boxes without visual noise.
- **Text & Hierarchy**: Primary body and critical data readouts use deep slate charcoal (`#1C2520`) for contrast ratio adherence against cream surfaces. Secondary labels, taxonomy markers, and table headers use muted slate (`#53635B`). Disabled or baseline helper text rests at `#7E9086`.
- **Instrumentation & Status**:
  - *Ready / Standby*: Sage green (`#52795D` fill, `#D8E0D8` border, `#1B3D2B` text).
  - *Inferring / Processing*: Warm terracotta / amber amber pulsing badge (`#C26D45` / `#D97706`).
  - *Completed / Converged*: Deep forest green (`#1B3D2B`).
  - *Error / Missing Checkpoint*: Brick red (`#A83832`).

## Typography

The type system pairs intellectual gravitas with rigorous data presentation:

- **Display & Section Titles (Newsreader)**: Used with restraint for app workspace branding, research publication titles, primary split headers, and modal title banners. It introduces editorial warmth, evoking scholarly papers and archival documentation.
- **System Interface & Controls (Inter)**: Handles parameter labels, long-form reconstruction notes, button text, and navigational links with neutral geometric clarity.
- **Instrument Readings & Metrics (JetBrains Mono)**: Used strictly for telemetry: PSNR, SSIM, LPIPS, routing tensors, seed numbers, inference execution latency (`142ms`), batch index counts, and raw parameter sliders. Always pair metric values with small uppercase labels using `label-caps`.

## Layout & Spacing

The layout is built on a high-density, analytical fixed/fluid hybrid structure:

- **Desktop (1280px+)**: Multi-column workbench layout. A fixed 320px left-hand control and model checkpoint rail, a flexible central dual-viewport canvas for side-by-side or split-slider tensor rendering, and a 360px collapsible right-hand inspection drawer for histogram, metrics, and diffusion step logs.
- **Tablet (768px - 1279px)**: The left configuration drawer shifts into a tabbed undercarriage sheet; the viewport maintains primary prominence with metric overlays displayed as floating HUD chips.
- **Mobile (<768px)**: Single-column linear layout. The inspection HUD condenses into a bottom-anchored swipe drawer; the canvas switches between 'Original' and 'Reconstructed' via a segmented pill toggle.
- **Rhythm**: All paddings and internal gaps scale across a crisp 4px/8px baseline rhythm (`space-xs` through `space-xl`), ensuring compact density suitable for data-heavy inspection views.

## Elevation & Depth

Visual hierarchy relies on structural boundaries and low-contrast surface contrast rather than drop shadows:

- **Tonal Layers**: The interface maintains three deliberate surface elevations:
  - *Tier 0 (Base Canvas)*: `#F9F8F3` (Ivory workspace foundation).
  - *Tier 1 (Panels & Toolbars)*: `#F4F2EB` (Recessed instrument docks, segmented switch tracks).
  - *Tier 2 (Interactive Viewports & Floating HUDs)*: `#FFFFFF` (Card surfaces, dropdown menus, context inspectors).
- **Outlines**: Structural definition is executed via crisp 1px borders using `#D8E0D8`. Active or hovered containers transition to `#6B8F71`.
- **Shadows**: Shadows are reserved strictly for floating viewport HUD overlays and transient popovers. When applied, use an ultra-diffused laboratory shadow tinted with forest green: `0 4px 20px -2px rgba(21, 51, 36, 0.06), 0 2px 6px -1px rgba(21, 51, 36, 0.04)`.

## Shapes

The design uses `Soft` roundedness (`roundedness: 1`):
- Small inputs, parameter chips, toggle handles, and data cells use a base `0.25rem` (4px) corner radius to evoke calibrated physical instruments.
- Panels, visual comparison canvases, and structural inspection cards use `rounded-lg` (`0.5rem` / 8px).
- Modals, large image lightboxes, and callout banners use `rounded-xl` (`0.75rem` / 12px).
- Pill radiuses are reserved exclusively for status badges (`Ready`, `Processing`) and comparison slider scrubber handles.

## Components

### Action Buttons
- **Primary Action (Run Inference / Restore)**: `#1B3D2B` background with `#F9F8F3` high-contrast text, 1px border of `#0F261A`, `0.25rem` radius. On hover: shifts to `#153324`. Active state: scale down to 0.98.
- **Secondary (Export Weights / Log State)**: `#FFFFFF` background with 1px border of `#D8E0D8`, text in `#1C2520`. Hover: `#F4F2EB` and `#6B8F71` border.
- **Tertiary / Utility**: Ghost styling with `#53635B` text; hover state introduces `#ECE8DD` background fill.

### Metric & Status Chips
- Height: 24px, padding: 0 8px. Font: `JetBrains Mono` at `0.75rem`.
- *Idle/Ready*: Background `#F4F2EB`, border 1px solid `#D8E0D8`, text `#1B3D2B`.
- *Inferring*: Background `rgba(194, 109, 69, 0.12)`, border 1px solid `#C26D45`, text `#C26D45`. Includes an animated 6px amber dot.
- *Completed*: Background `rgba(82, 121, 93, 0.12)`, border 1px solid `#52795D`, text `#1B3D2B`.

### Form Controls & Parameter Sliders
- **Numeric & Seed Inputs**: `#FFFFFF` background, 1px `#D8E0D8` border, monospace data presentation. Focus state triggers a 1px border in `#1B3D2B` without outer glow.
- **Sliders (Denoising Steps, Guidance Scale)**: Track height: 4px in `#D8E0D8`; active progress bar in `#1B3D2B`. Scrubber: 14px solid square with 2px radius in `#FFFFFF`, enveloped by a 2px `#1B3D2B` border.

### Image Viewports & Inspection Panels
- Double-buffered canvas container featuring a 1px solid border (`#D8E0D8`) on `#FCFBF9`.
- Integrated split-slider scrubber with vertical 1px hairline divider (`#FFFFFF`) with a dual-arrow circular tactile knob.
- Bottom telemetry strip within the canvas: overlays Latency, Resolution, Seed, and Memory consumption rendered in `metric-sm` using translucent white backdrops (`rgba(255, 255, 255, 0.88)` with `backdrop-filter: blur(8px)`).

### Metric Comparison Cards
- Surface: `#FFFFFF`, border: 1px solid `#D8E0D8`.
- Layout: Top line contains `label-caps` in `#53635B` (e.g., `PSNR / SSIM EVALUATION`).
- Value section: Large monospace readout in `metric-lg` (`#1C2520`) with delta tag adjacent (e.g., `+3.42 dB` in `#52795D`).