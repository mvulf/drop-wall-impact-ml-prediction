# Plan: Split `submit2_K_comparison.pdf` (vulf15.pdf) into 2 figures

## Current state

The script saves a single figure `submit2_K_comparison.pdf` which is then copied to `Dissertation/images/vulf15.pdf`.

The figure is a **4 columns x 3 rows** grid (12 subplots, labeled a-l):

```
         Wettability          |        Inclination
      Splashing  No-frag      |     Splashing  No-frag
Stk:    (a)       (b)         |       (c)       (d)
d_p:    (e)       (f)         |       (g)       (h)
phi:    (i)       (j)         |       (k)       (l)
```

- Columns 1-2 (a,b,e,f,i,j): **Wettability** influence (legend: lyophilic / neutral / lyophobic, solid lines)
- Columns 3-4 (c,d,g,h,k,l): **Inclination** influence (legend: 0 / 20 / 45 degrees, dash-dot lines)
- Odd columns (1,3): **Splashing** hyperplanes
- Even columns (2,4): **No fragmentation** hyperplanes
- Row 1: Y-axis = Stk_sed (log scale)
- Row 2: Y-axis = d_p / D_drop (log scale)
- Row 3: Y-axis = phi_drop (linear scale)
- All X-axes: K-parameter (0..250)
- Colors: blue, green, red (same for both halves)

## Target: 2 separate figures

### Figure 1: `submit2_K_comparison_wettability.pdf` -> `vulf15a.pdf`

Grid **2 columns x 3 rows** (6 subplots, labeled **a-f**):

```
      Splashing  No-frag
Stk:    (a)       (b)
d_p:    (c)       (d)
phi:    (e)       (f)
```

- Content = current columns 1-2 (wettability)
- Legend: lyophilic / neutral / lyophobic (solid lines, colors: blue/green/red)

### Figure 2: `submit2_K_comparison_inclination.pdf` -> `vulf15b.pdf`

Grid **2 columns x 3 rows** (6 subplots, labeled **a-f**):

```
      Splashing  No-frag
Stk:    (a)       (b)
d_p:    (c)       (d)
phi:    (e)       (f)
```

- Content = current columns 3-4 (inclination)
- Legend: 0 / 20 / 45 degrees (dash-dot lines, colors: blue/green/red)

## Python script changes

1. **Split the figure generation** into two separate `fig, axes = plt.subplots(3, 2, ...)` calls instead of one `plt.subplots(3, 4, ...)`.

2. **Recommended figsize**: `(8, 10)` or `(9, 10)` for each (was likely ~`(16, 10)` for the combined figure). This makes each subplot ~2x wider than before.

3. **Font sizes** (increase for readability):
   - Axis labels: 14 pt
   - Tick labels: 12 pt
   - Legend: 12 pt
   - Subplot letters (a), (b), etc.: 14 pt bold

4. **Subplot labels** must be re-lettered **(a) through (f)** for each figure independently.

5. **Save** as two separate PDFs:
   - `submit2_K_comparison_wettability.pdf`
   - `submit2_K_comparison_inclination.pdf`

6. **Copy** to dissertation:
   - `submit2_K_comparison_wettability.pdf` -> `Dissertation/images/vulf15a.pdf`
   - `submit2_K_comparison_inclination.pdf` -> `Dissertation/images/vulf15b.pdf`

## LaTeX changes in `Dissertation/part5.tex`

### 1. Replace the figure block (lines 1029-1039)

**Old** (single figure, 12 subplots a-l):
```latex
\begin{figure*}[htpb]
	\includegraphics[width=0.98\textwidth]{vulf15.pdf}
	\caption{
		\label{fig_model_hyperplanes}
		Features influence on separating hyperplane.
		\textit{Splashing} hyperplanes are introduced in the first and third plot columns, (a, e, i) and (c, g, k), respectively.
		\textit{No fragmentation} hyperplanes - the second and fourth plot columns, (b, f, j) and (d, h, l), respectively.
		\textit{Wettability} influence is introduced in the first two columns (a-b, e-f, i-j).
		\textit{Inclination} - the last two columns (c-d, g-h, k-l).
	}
\end{figure*}
```

**New** (two figures, 6 subplots a-f each):
```latex
\begin{figure*}[htpb]
	\centering
	\includegraphics[width=0.98\textwidth]{vulf15a.pdf}
	\caption{
		\label{fig_model_hyperplanes_wettability}
		Effect of substrate \textit{wettability} on separating hyperplane.
		\textit{Splashing} hyperplanes~-- (a, c, e).
		\textit{No fragmentation} hyperplanes~-- (b, d, f).
	}
\end{figure*}

\FloatBarrier

\begin{figure*}[htpb]
	\centering
	\includegraphics[width=0.98\textwidth]{vulf15b.pdf}
	\caption{
		\label{fig_model_hyperplanes_inclination}
		Effect of substrate \textit{inclination} on separating hyperplane.
		\textit{Splashing} hyperplanes~-- (a, c, e).
		\textit{No fragmentation} hyperplanes~-- (b, d, f).
	}
\end{figure*}
```

### 2. Update all text references (7 locations)

**Line 1026** (intro paragraph):
```
OLD: ...effect of substrate \textit{wettability} (\Fig~\ref{fig_model_hyperplanes}, first two plot columns) and the substrate \textit{inclination} angle (\Fig~\ref{fig_model_hyperplanes}, last two plot columns)...
NEW: ...effect of substrate \textit{wettability} (\Fig~\ref{fig_model_hyperplanes_wettability}) and the substrate \textit{inclination} angle (\Fig~\ref{fig_model_hyperplanes_inclination})...
```

**Line 1043** (wettability result):
```
OLD: ...(\Fig~\ref{fig_model_hyperplanes}, first two plot columns).
NEW: ...(\Fig~\ref{fig_model_hyperplanes_wettability}).
```

**Line 1044** (inclination result):
```
OLD: ...(\Fig~\ref{fig_model_hyperplanes}, last two plot columns).
NEW: ...(\Fig~\ref{fig_model_hyperplanes_inclination}).
```

**Line 1047** (sensitivity comparison):
```
OLD: ...variations in the \textit{wettability} and \textit{inclination} (\Fig~\ref{fig_model_hyperplanes}, second and fourth plot columns) than the \textit{splashing} classifier (\Fig~\ref{fig_model_hyperplanes}, first and third plot columns).
NEW: ...variations in the \textit{wettability} and \textit{inclination} (\Fig~\ref{fig_model_hyperplanes_wettability}(b, d, f) and \Fig~\ref{fig_model_hyperplanes_inclination}(b, d, f)) than the \textit{splashing} classifier (\Fig~\ref{fig_model_hyperplanes_wettability}(a, c, e) and \Fig~\ref{fig_model_hyperplanes_inclination}(a, c, e)).
```

**Line 1052** (phi_drop discussion):
```
OLD: Let us consider the last line of the plots in \Fig~\ref{fig_model_hyperplanes}, where...
NEW: Let us consider the last row of plots in \Fig~\ref{fig_model_hyperplanes_wettability} and \Fig~\ref{fig_model_hyperplanes_inclination}, where...
```

**Line 1053** (subplot letter references):
```
OLD: ...for \textit{fragmentation} (\Fig~\ref{fig_model_hyperplanes}(j,l)) than for splashing (\Fig~\ref{fig_model_hyperplanes}(i,k)).
NEW: ...for \textit{fragmentation} (\Fig~\ref{fig_model_hyperplanes_wettability}(f) and \Fig~\ref{fig_model_hyperplanes_inclination}(f)) than for splashing (\Fig~\ref{fig_model_hyperplanes_wettability}(e) and \Fig~\ref{fig_model_hyperplanes_inclination}(e)).
```

## Subplot letter mapping (old -> new)

### Wettability figure (vulf15a.pdf):
| Old letter | New letter | Content |
|---|---|---|
| (a) | **(a)** | Splashing, Stk_sed |
| (b) | **(b)** | No-frag, Stk_sed |
| (e) | **(c)** | Splashing, d_p/D_drop |
| (f) | **(d)** | No-frag, d_p/D_drop |
| (i) | **(e)** | Splashing, phi_drop |
| (j) | **(f)** | No-frag, phi_drop |

### Inclination figure (vulf15b.pdf):
| Old letter | New letter | Content |
|---|---|---|
| (c) | **(a)** | Splashing, Stk_sed |
| (d) | **(b)** | No-frag, Stk_sed |
| (g) | **(c)** | Splashing, d_p/D_drop |
| (h) | **(d)** | No-frag, d_p/D_drop |
| (k) | **(e)** | Splashing, phi_drop |
| (l) | **(f)** | No-frag, phi_drop |

## Checklist

- [ ] Modify Python script: split into 2 figures (2x3 each)
- [ ] Re-letter subplots (a)-(f) in each figure
- [ ] Increase font sizes (axes: 14pt, ticks: 12pt, legend: 12pt)
- [ ] Save as `submit2_K_comparison_wettability.pdf` and `submit2_K_comparison_inclination.pdf`
- [ ] Copy to `Dissertation/images/vulf15a.pdf` and `vulf15b.pdf`
- [ ] Update LaTeX: replace figure block with two figures
- [ ] Update LaTeX: fix all 7 text references (lines 1026, 1043, 1044, 1047, 1052, 1053)
- [ ] Verify no other references to `fig_model_hyperplanes` remain
- [ ] Compile and check numbering (Figure 4.16 -> wettability, Figure 4.17 -> inclination)
