---
name: Predictive Precision System
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#434655'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#737686'
  outline-variant: '#c3c6d7'
  surface-tint: '#0053db'
  primary: '#004ac6'
  on-primary: '#ffffff'
  primary-container: '#2563eb'
  on-primary-container: '#eeefff'
  inverse-primary: '#b4c5ff'
  secondary: '#006c49'
  on-secondary: '#ffffff'
  secondary-container: '#6cf8bb'
  on-secondary-container: '#00714d'
  tertiary: '#4d556b'
  on-tertiary: '#ffffff'
  tertiary-container: '#656d84'
  on-tertiary-container: '#eef0ff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dbe1ff'
  primary-fixed-dim: '#b4c5ff'
  on-primary-fixed: '#00174b'
  on-primary-fixed-variant: '#003ea8'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#dae2fd'
  tertiary-fixed-dim: '#bec6e0'
  on-tertiary-fixed: '#131b2e'
  on-tertiary-fixed-variant: '#3f465c'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  headline-xl:
    fontFamily: Plus Jakarta Sans
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.025em
  headline-xl-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 30px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.005em
  label-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0em
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.04em
  metric-display:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 38px
    letterSpacing: -0.02em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-mobile: 1rem
  margin: 2rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
---

## Brand & Style

The design system is engineered for high-stakes forecasting, academic rigor, and enterprise inventory intelligence. Balancing scientific clarity with high-performance SaaS usability, the visual aesthetic leans strictly into **Modern Functional Minimalism** with refined corporate precision.

### Target Audience & Emotional Arc
- **Users**: Data scientists, retail demand planners, supply chain researchers, and enterprise inventory directors analyzing multi-layer perceptron (MLP) outputs.
- **Emotional Response**: The interface communicates statistical authority, computational reliability, uncompromised signal-to-noise ratio, and operational control. Visual friction, noise, and superficial ornamentation are eliminated.

### Visual Aesthetic
- Clean architectural layouts grounded on high-contrast data visualization.
- Subtle geometric structure governed by strict horizontal and vertical alignments.
- Pure white analysis cards anchored on cool slate neutral fields, treating tabular and metric data as first-class visual citizens.

## Colors

The palette establishes an analytical hierarchy through cold slates, functional blues, and decisive precision accents:

### System Roles & Palette Structure
- **Primary (`#2563EB` / Active: `#1D4ED8`)**: Used for primary user intents, key model actions, forecast execution controls, and active series in chart visualizers.
- **Secondary / Signal Positive (`#10B981` / Active: `#059669`)**: Reserved strictly for high-accuracy markers, positive delta performance, validation metrics (RMSE/RMSLE reductions), and optimal stock states.
- **Structure & Depth (`#0F172A`, `#1E293B`)**: Serves as the bedrock for structural headings, high-priority metric values, navigation chrome, and severe delta labels.
- **Surfaces & Canvases (`#F8FAFC`, `#F1F5F9`)**: Cool, desaturated base canvases that mitigate eye fatigue during long analytical sessions while creating separation from elevated white data cards (`#FFFFFF`).
- **Dividers & Structural Borders (`#E2E8F0`, `#CBD5E1`)**: Crisp, low-weight structural outlines defining tables, panels, metric boundaries, and data partitions without visual heaviness.
- **Critical & Error State (`#EF4444`)**: Deployed exclusively for model convergence failures, out-of-stock alarms, and data anomaly warnings.

## Typography

The dual typographic strategy integrates **Plus Jakarta Sans** for structural headers and KPI callouts with **Inter** for dense numerical tables, parameter lists, and analytical documentation.

- **Tabular Figures**: Tabular numerals (`font-variant-numeric: tabular-nums`) are mandatory across all numerical instances, including loss histories, confidence intervals, item IDs, dates, and inventory units.
- **Hierarchy & Case**: Small statistical tags and table headers enforce `label-sm` with subtle uppercase tracking to ensure distinct contrast against actual cell data.
- **Reading Comfort**: Body copy preserves an exact 1.45–1.5 line height ratio to guarantee scanning precision across long training reports and model parameter cards.

## Layout & Spacing

The layout is built upon an 8pt architectural rhythm, utilizing a structured 12-column fluid grid system on desktop that scales gracefully down to single-column workflows on constrained viewports.

### Grid Anatomy & Breakpoints
- **Desktop (≥1280px)**: 12 columns, 24px (`1.5rem`) gutters, 32px (`2rem`) canvas margins. Max layout width constrained to 1600px for large monitoring dashboards.
- **Tablet (768px – 1279px)**: 8 columns, 16px (`1rem`) gutters, 24px (`1.5rem`) margins. Side-by-side metric tiles collapse into 2x2 grids.
- **Mobile (<768px)**: 4 columns, 16px (`1rem`) gutters, 16px (`1rem`) canvas margins. Multi-metric evaluation tables adopt horizontal scroll containers with frozen primary ID columns.

### Density & Rhythms
- **Compact Data Groupings**: `space-xs` (4px) and `space-sm` (8px) govern input internals, table cell padding, and metadata key-value separations.
- **Section & Module Gaps**: `space-lg` (24px) separates card panels; `space-xl` (32px) isolates major dashboard sections (e.g., Model Topology from Evaluation Results).

## Elevation & Depth

This design system uses a **low-contrast outline + ambient micro-shadow** philosophy to keep numerical content crisp, sharp, and flat. Heavy blurs and theatrical drop shadows are prohibited.

### Elevation Architecture
1. **Level 0 (Base Canvas)**: Deep slate neutral background (`#F8FAFC`). No shadow, no border.
2. **Level 1 (Data Cards & Analysis Panels)**: Surface `#FFFFFF`, bounded by a 1px solid `#E2E8F0` border, augmented by an ambient diffusion: `0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02)`.
3. **Level 2 (Hover States, Popovers & Contextual Drawers)**: Surface `#FFFFFF`, 1px solid `#CBD5E1`, with focused micro-depth: `0 4px 6px -1px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.04)`.
4. **Level 3 (Modal Dialogs & Predictive Scenario Overlays)**: Surface `#FFFFFF`, accompanied by an overlay mask of `#0F172A` at 40% opacity, paired with an elevation shadow of `0 20px 25px -5px rgba(15, 23, 42, 0.1), 0 8px 10px -6px rgba(15, 23, 42, 0.04)`.

## Shapes

The shape system adopts a **Soft (`1`)** geometry. Crisp angles with subtle curvature create an analytical, clean, and disciplined instrument feel, avoiding both severe raw brutalism and overly consumer-facing roundedness.

- **Base Radius (`0.25rem` / 4px)**: Applied to table cells, tags, micro-badges, inputs, code snippets, and button components.
- **Card & Container Radius (`0.5rem` / 8px)**: Applied to evaluation cards, prediction parameter panels, chart modules, and metric containers.
- **Modal & Drawer Radius (`0.75rem` / 12px)**: Applied to global dialogs and major overlay drawers.

## Components

### Buttons
- **Primary**: Background `#2563EB`, text `#FFFFFF`, border none, hover `#1D4ED8`, active `#1E40AF`. Font `label-md`, height 36px, horizontal padding 16px, radius 4px.
- **Secondary / Outline**: Background `#FFFFFF`, text `#0F172A`, 1px solid border `#CBD5E1`, hover background `#F8FAFC`.
- **Tertiary / Ghost**: Background transparent, text `#64748B`, hover text `#0F172A`, hover background `#F1F5F9`.

### Chips & Status Indicators
- **Metric Chips**: Padding 2px 8px, radius 4px, font `label-sm`.
  - *Optimal / Validated*: Background `#ECFDF5`, text `#059669`, border 1px solid `#A7F3D0`.
  - *In-Progress / Epoch*: Background `#EFF6FF`, text `#1D4ED8`, border 1px solid `#BFDBFE`.
  - *Anomaly / Outlier*: Background `#FEF2F2`, text `#B91C1C`, border 1px solid `#FECACA`.

### Lists & Evaluation Tables
- **Header Row**: Height 40px, background `#F8FAFC`, bottom border 1px solid `#E2E8F0`, text `label-sm` in `#64748B` with uppercase tracking.
- **Data Rows**: Height 44px, bottom border 1px solid `#F1F5F9`, hover background `#F8FAFC`. Numbers formatted via monospace-aligned tabular figures.
- **Selection Rows**: Background `#EFF6FF` with a 2px left accent border `#2563EB`.

### Checkboxes & Radio Buttons
- 16x16px bounding box with 1px border `#CBD5E1`.
- Checked state: `#2563EB` fill with a white 1.5px checkmark/radio dot. Focus ring: 2px offset with `#93C5FD`.

### Input Fields & Parameter Selectors
- **Input Height**: 36px. Border 1px solid `#CBD5E1`, surface `#FFFFFF`, text `body-md` in `#0F172A`, placeholder in `#94A3B8`.
- **Focus**: Border `#2563EB` with an ambient glow outline: `box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15)`.

### Cards & Metric Panels
- **KPI Card**: Minimum height 112px, 16px internal padding. Displays the metric name (`label-sm`, `#64748B`), the big metric value (`metric-display`, `#0F172A`), and the confidence/delta delta badge (`label-sm`).
- **Graph Containers**: White panel bounded by `#E2E8F0` with a standardized header slot containing chart toggles (e.g., actual vs. predicted MLP curves) and an export action trigger.

### Specialized AI & Forecasting Components
- **Epoch Training Progress Bar**: 4px track height, surface `#E2E8F0`, indicator bar `#2563EB` with smooth transition easing.
- **Confidence Interval Band**: Multi-layered fill chart component using `#2563EB` at 10% opacity for 95% confidence intervals, bounded by a 1px dashed reference border.