---
name: IPB Calibration Report
description: Light, calm engineering report on a soft blue gradient; white-glass cards, one blue accent, tabular numbers.
colors:
  ink: "#1b2430"
  ink-muted: "#334050"
  accent: "#2358c2"
  accent-deep: "#1c47a0"
  card: "rgba(255,255,255,.62)"
  line: "rgba(27,36,48,.14)"
  page-from: "#92B0E8"
  page-to: "#EEEEEE"
  viewer-bg: "#0d1117"
  camera-marker: "#2f6fdb"
  lidar-marker: "#e07b24"
  viewer-text: "#cccccc"
  map-grey: "#8a8f98"
typography:
  display:
    fontFamily: "Jost, system-ui, sans-serif"
    fontSize: "46px"
    fontWeight: 600
    lineHeight: 1.1
  headline:
    fontFamily: "Jost, system-ui, sans-serif"
    fontSize: "32px"
    fontWeight: 600
    lineHeight: 1.15
  title:
    fontFamily: "Jost, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 600
  body:
    fontFamily: "Jost, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.4
  kpi:
    fontFamily: "Jost, system-ui, sans-serif"
    fontSize: "30px"
    fontWeight: 600
    lineHeight: 1.2
  label:
    fontFamily: "Jost, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 500
  data:
    fontFamily: "ui-monospace, Menlo, monospace"
    fontSize: "11px"
    lineHeight: 1.4
rounded:
  control: "8px"
  card: "12px"
  slide: "14px"
spacing:
  sp-1: "8px"
  sp-2: "12px"
  sp-3: "16px"
  sp-4: "24px"
  sp-5: "32px"
components:
  card:
    backgroundColor: "{colors.card}"
    textColor: "{colors.ink}"
    rounded: "{rounded.card}"
    padding: "12px 16px"
  kpi:
    backgroundColor: "{colors.card}"
    textColor: "{colors.accent}"
    rounded: "{rounded.card}"
    padding: "12px 8px"
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "#ffffff"
    rounded: "{rounded.control}"
    padding: "10px 16px"
  button-primary-hover:
    backgroundColor: "{colors.accent-deep}"
  tab-button:
    backgroundColor: "{colors.card}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "6px 12px"
  tab-button-active:
    backgroundColor: "{colors.accent}"
    textColor: "#ffffff"
---

# Design System: IPB Calibration Report

Source of truth: `templates/report/run_report_template.html` (single run) and `templates/report/slides_template.html` (experiment deck). Both share this system.

## Overview

**Creative North Star: "The Lab Notebook, Printed on Sky"**

A calm, light report an engineer reads once to decide whether a calibration is trustworthy. The page is a soft blue-to-grey gradient; content sits on translucent white cards. Numbers are the content: tabular figures, generous cell padding, one accent colour reserved for headline values and active controls. No decoration beyond the gradient, the card glass and the dark 3D viewer, which is the only dark surface.

**Key Characteristics:**
- Fluid sections: full width up to 1280 px, KPIs/rows/image grid wrap, wide tables scroll inside their card. Print = A4 landscape, sections flow across pages (no forced page break per slide; cards, KPIs, figures and table rows are kept whole).
- One accent blue; camera/LiDAR markers (blue/orange) appear only in plots.
- Headings carry the structure: no kickers, no section numbers.
- Light only, no dark mode.

## Colors

Cool, low-chroma neutrals plus one saturated blue.
- **Ink** `#1b2430` body and headings; **Ink muted** `#334050` captions, table heads (≥ 7:1 on the card).
- **Accent** `#2358c2` KPI values, primary and active buttons; **Accent deep** on hover.
- **Page gradient** `radial-gradient(at left bottom, #92B0E8 2%, #EEEEEE 100%)`, on `html` and on every slide.
- **Card** `rgba(255,255,255,.62)` with a `rgba(27,36,48,.14)` 1 px line.
- **Viewer** `#0d1117` behind the 3D cloud. LiDAR palette in the viewer: `#ff5964 #ffb140 #ffe66d #7bd389 #35c4d8 #4d8af0 #a06cf0 #f06cc8`; legend swatches are dots, label text stays ink (yellow text fails contrast on light cards).

## Typography

Jost throughout (Google Fonts, system fallback offline). Monospace only for `report.txt` and metric meanings.
- Display 46/1.1 for the run name; headline 32/1.15 per slide; title 18 for card headings; body 17/1.4; labels, captions, table text 14.
- `font-variant-numeric: tabular-nums` on tables and KPIs. Numbers over 999 use thousand separators (`1,432`).
- Body measure capped at 75ch, captions at 110ch.

## Layout

- Spacing scale: 8 / 12 / 16 / 24 / 32 (`--sp-1..5`). **Only `gap` separates siblings**: each slide is a flex column with `gap: 16px`, cards in a row use 16, KPI and control rows 8-12. No margin-top on children, no `margin-top:auto`.
- Slide padding 32 / 48 / 40 (top / sides / bottom); slides are separated by 32 px; slide height follows content (no empty bottom). Padding and headings scale with `clamp()`.
- Order inside a slide: heading, optional meta/lead, content card(s), caption.
- Tables sit in a card, header muted, 7 × 10 px cells, last row has no rule; first two columns left-aligned, numbers right-aligned.
- Interactive plots: images in a 2-column grid (1 column when a single view is chosen); 3D viewer fixed 560 px high.

## Elevation & Depth

Flat tonal layering. Cards have no shadow (translucent fill + hairline). Only the slide itself lifts off the page (`0 6px 28px rgba(27,36,48,.22)`) and the floating Export button (`0 2px 8px rgba(0,0,0,.2)`). Print removes shadows and radii.

## Shapes

Radius 8 (buttons, selects, images, 3D viewer), 12 (cards, KPIs), 14 (slides). No pills, no coloured side borders.

## Components

- **Card**: glass surface described above; never nested inside another card (tables and plots are the card's direct content).
- **KPI tile**: accent value 30 px (`kpi` role), 14 px label, two rows on the cover (counts, then quality metrics).
- **Tab controls** (`.tabs`): selects, buttons, range sliders; min height 36 px, active state filled accent, focus ring 3 px ink with 2 px offset.
- **Export PDF**: fixed top-right, primary button style, hidden in print.

## Do's and Don'ts

- Do keep all numbers tabular and separated; do state units in column heads.
- Do space with the scale and `gap`; do check there is no void below the last element of a slide.
- Do keep coloured text off light cards unless it is the accent.
- Don't add kickers/eyebrows, section numbers, gradient text, glass-on-glass or nested cards.
- Don't encode meaning in colour alone (camera ● vs LiDAR ■ also differ by shape).
- Don't fix slide widths or heights; content decides.
