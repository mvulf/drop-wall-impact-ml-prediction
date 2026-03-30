# Plan: Split vulf14.pdf (submit2_K_classification.pdf) into 4 figures

## Current state

The original figure `vulf14.pdf` (generated as `submit2_K_classification.pdf`) is a **4x4 grid** of 16 subplots:

```
         Left half (Stk, phi)           Right half (dp/D, Ra/D)
       Col1(LR)   Col2(CB)           Col3(LR)   Col4(CB)
Row1:   (a)         (b)                (c)         (d)        <- Splashing
Row2:   (e)         (f)                (g)         (h)        <- Fragmentation
Row3:   (i)         (j)                (k)         (l)        <- Splashing
Row4:   (m)         (n)                (o)         (p)        <- Fragmentation
```

Axes mapping:
- Rows 1-2, y-axis: `Stk_sed` (left) and `d_p/D_drop` (right)
- Rows 3-4, y-axis: `phi_drop` (left) and `Ra/D_drop` (right)
- All x-axes: K-parameter

Columns mapping:
- Col 1, 3: Logistic Regression
- Col 2, 4: CatBoost

Rows mapping:
- Row 1, 3: Splashing (3-class legend: no splashing, semi splashing, splashing)
- Row 2, 4: No fragmentation (2-class legend: fragmentation, no fragmentation)

## Target: 4 separate 2x2 figures

### Figure 1: Stk_sed vs K-parameter
File: `vulf14a.pdf`

```
       Col1(LR)    Col2(CB)
Row1:   (a)          (b)     <- Splashing      (y: Stk_sed)
Row2:   (c)          (d)     <- Fragmentation  (y: Stk_sed)
```

Old letters -> New letters:
- (a) -> (a)
- (b) -> (b)
- (e) -> (c)
- (f) -> (d)

### Figure 2: d_p/D_drop vs K-parameter
File: `vulf14b.pdf`

```
       Col1(LR)    Col2(CB)
Row1:   (a)          (b)     <- Splashing      (y: d_p/D_drop)
Row2:   (c)          (d)     <- Fragmentation  (y: d_p/D_drop)
```

Old letters -> New letters:
- (c) -> (a)
- (d) -> (b)
- (g) -> (c)
- (h) -> (d)

### Figure 3: phi_drop vs K-parameter
File: `vulf14c.pdf`

```
       Col1(LR)    Col2(CB)
Row1:   (a)          (b)     <- Splashing      (y: phi_drop)
Row2:   (c)          (d)     <- Fragmentation  (y: phi_drop)
```

Old letters -> New letters:
- (i) -> (a)
- (j) -> (b)
- (m) -> (c)
- (n) -> (d)

### Figure 4: Ra/D_drop vs K-parameter
File: `vulf14d.pdf`

```
       Col1(LR)    Col2(CB)
Row1:   (a)          (b)     <- Splashing      (y: Ra/D_drop)
Row2:   (c)          (d)     <- Fragmentation  (y: Ra/D_drop)
```

Old letters -> New letters:
- (k) -> (a)
- (l) -> (b)
- (o) -> (c)
- (p) -> (d)

---

## Changes to Python script (submit2_K_classification.pdf generator)

The script currently builds one `fig` with a 4x4 grid.
Need to refactor so it produces 4 separate PDF files, each with a 2x2 grid.

### What to change in the script:

1. **Replace the single 4x4 figure** with a loop over 4 y-variables:
   - `Stk_sed`, `d_p/D_drop`, `phi_drop`, `Ra/D_drop`

2. **For each y-variable**, create a `fig, axes = plt.subplots(2, 2, ...)`:
   - Row 0: Splashing (LR, CatBoost)
   - Row 1: Fragmentation (LR, CatBoost)

3. **Subplot labels**: each figure uses `(a)`, `(b)`, `(c)`, `(d)` independently.

4. **Figure size**: use approximately `figsize=(10, 8)` or similar (half the original width, half the original height) to ensure readable fonts and markers.

5. **Save** each figure separately:
   - `submit2_K_classification_stk.pdf` (-> copy to `vulf14a.pdf`)
   - `submit2_K_classification_dp.pdf` (-> copy to `vulf14b.pdf`)
   - `submit2_K_classification_phi.pdf` (-> copy to `vulf14c.pdf`)
   - `submit2_K_classification_ra.pdf` (-> copy to `vulf14d.pdf`)

6. **Legends**: each figure keeps the same legend style as before (top row: 3-class splashing, bottom row: 2-class fragmentation).

7. **Colorbars**: one shared colorbar per figure (right side), same as original.

8. **Font sizes**: increase all font sizes since the figure is now 2x2 instead of 4x4. Target ~12-14pt for axis labels, ~10-12pt for tick labels.

---

## Changes to LaTeX (Dissertation/part5.tex)

### Replace single figure block (lines 955-966) with 4 figure blocks:

**BEFORE** (lines 955-966):
```latex
% TODO: EDIT THIS PLOT AND BELOW
\begin{figure*}[!ht]
    \includegraphics[width=1.0\textwidth]{vulf14.pdf}
    \caption{
        \label{fig_model_predictions}
        Prediction for the horizontal lyophilic smooth substrate.
        Logistic Regression is introduced in the first and third plot columns, (a, e, i, m) and (c, g, k, o), respectively.
        CatBoost - second and fourth plot columns, (b, f, j, n) and (d, h, l, p), respectively.
        \textit{Splashing} predictions are introduced in the first and third plot rows, (a-d) and (i-l), respectively.
        \textit{No fragmentation} predictions - the second and fourth plot rows, (e-h) and (m-p), respectively.
    }
\end{figure*}
```

**AFTER:**
```latex
\begin{figure*}[!ht]
    \centering
    \includegraphics[width=1.0\textwidth]{vulf14a.pdf}
    \caption{
        \label{fig_model_predictions_stk}
        Prediction for the horizontal lyophilic smooth substrate: effect of \textit{sedimentation Stokes number} $\text{Stk}_\text{sed}$.
        Logistic Regression -- (a, c), CatBoost -- (b, d).
        \textit{Splashing} predictions -- (a, b), \textit{no fragmentation} predictions -- (c, d).
    }
\end{figure*}

\begin{figure*}[!ht]
    \centering
    \includegraphics[width=1.0\textwidth]{vulf14b.pdf}
    \caption{
        \label{fig_model_predictions_dp}
        Prediction for the horizontal lyophilic smooth substrate: effect of \textit{particle-droplet diameter ratio} $d_\text{p}/D_\text{drop}$.
        Logistic Regression -- (a, c), CatBoost -- (b, d).
        \textit{Splashing} predictions -- (a, b), \textit{no fragmentation} predictions -- (c, d).
    }
\end{figure*}

\begin{figure*}[!ht]
    \centering
    \includegraphics[width=1.0\textwidth]{vulf14c.pdf}
    \caption{
        \label{fig_model_predictions_phi}
        Prediction for the horizontal lyophilic smooth substrate: effect of \textit{volume fraction} $\phi_\text{drop}$.
        Logistic Regression -- (a, c), CatBoost -- (b, d).
        \textit{Splashing} predictions -- (a, b), \textit{no fragmentation} predictions -- (c, d).
    }
\end{figure*}

\begin{figure*}[!ht]
    \centering
    \includegraphics[width=1.0\textwidth]{vulf14d.pdf}
    \caption{
        \label{fig_model_predictions_ra}
        Prediction for the horizontal lyophilic smooth substrate: effect of \textit{relative roughness} $\text{Ra}/D_\text{drop}$.
        Logistic Regression -- (a, c), CatBoost -- (b, d).
        \textit{Splashing} predictions -- (a, b), \textit{no fragmentation} predictions -- (c, d).
    }
\end{figure*}
```

### Update references in part5.tex text (lines 979-997):

| Line | Old reference | New reference |
|------|--------------|---------------|
| 979 | `\Fig~\ref{fig_model_predictions}` | `\Fig~\ref{fig_model_predictions_stk}--\ref{fig_model_predictions_ra}` |
| 987 | `\Fig~\ref{fig_model_predictions}(a,e)` | `\Fig~\ref{fig_model_predictions_stk}(a,c)` |
| 988 | `\Fig~\ref{fig_model_predictions}(c,g)` | `\Fig~\ref{fig_model_predictions_dp}(a,c)` |
| 994 | `\Fig~\ref{fig_model_predictions}` | `\Fig~\ref{fig_model_predictions_stk}--\ref{fig_model_predictions_ra}` |
| 997 | `\Fig~\ref{fig_model_predictions}` | `\Fig~\ref{fig_model_predictions_stk}--\ref{fig_model_predictions_ra}` |

### Update references in appendix.tex (lines 503-527):

| Line | Old reference | New reference |
|------|--------------|---------------|
| 503 | `\Fig~\ref{fig_model_predictions}` | `\Fig~\ref{fig_model_predictions_stk}--\ref{fig_model_predictions_ra}` |
| 514 | `\Fig~\ref{fig_model_predictions}` | `\Fig~\ref{fig_model_predictions_stk}--\ref{fig_model_predictions_ra}` |
| 517 | `\Fig~\ref{fig_model_predictions}(k,l,o,p)` | `\Fig~\ref{fig_model_predictions_ra}` |
| 520 | `\Fig~\ref{fig_model_predictions}(i,j,m,n)` | `\Fig~\ref{fig_model_predictions_phi}` |
| 521 | `\Fig~\ref{fig_model_predictions}(a, b, e, f)` | `\Fig~\ref{fig_model_predictions_stk}` |
| 522 | `\Fig~\ref{fig_model_predictions}(c, d, g, h)` | `\Fig~\ref{fig_model_predictions_dp}` |
| 523 | `\Fig~\ref{fig_model_predictions}(i-p)` | `\Fig~\ref{fig_model_predictions_phi}, \ref{fig_model_predictions_ra}` |
| 527 | `\Fig~\ref{fig_model_predictions}` | `\Fig~\ref{fig_model_predictions_stk}--\ref{fig_model_predictions_ra}` |

---

## Old-to-new letter mapping (complete)

For reference when updating any text that mentions specific subplot letters:

| Old (4x4) | New figure | New letter | Description |
|------------|-----------|------------|-------------|
| (a) | vulf14a (Stk) | (a) | LR, Splashing, Stk_sed |
| (b) | vulf14a (Stk) | (b) | CB, Splashing, Stk_sed |
| (c) | vulf14b (dp) | (a) | LR, Splashing, d_p/D |
| (d) | vulf14b (dp) | (b) | CB, Splashing, d_p/D |
| (e) | vulf14a (Stk) | (c) | LR, Fragmentation, Stk_sed |
| (f) | vulf14a (Stk) | (d) | CB, Fragmentation, Stk_sed |
| (g) | vulf14b (dp) | (c) | LR, Fragmentation, d_p/D |
| (h) | vulf14b (dp) | (d) | CB, Fragmentation, d_p/D |
| (i) | vulf14c (phi) | (a) | LR, Splashing, phi_drop |
| (j) | vulf14c (phi) | (b) | CB, Splashing, phi_drop |
| (k) | vulf14d (Ra) | (a) | LR, Splashing, Ra/D |
| (l) | vulf14d (Ra) | (b) | CB, Splashing, Ra/D |
| (m) | vulf14c (phi) | (c) | LR, Fragmentation, phi_drop |
| (n) | vulf14c (phi) | (d) | CB, Fragmentation, phi_drop |
| (o) | vulf14d (Ra) | (c) | LR, Fragmentation, Ra/D |
| (p) | vulf14d (Ra) | (d) | CB, Fragmentation, Ra/D |

---

## Checklist

- [ ] Modify Python script: split 4x4 figure into 4 separate 2x2 figures
- [ ] Generate 4 PDFs: vulf14a.pdf, vulf14b.pdf, vulf14c.pdf, vulf14d.pdf
- [ ] Copy PDFs to `Dissertation/images/`
- [ ] Update part5.tex: replace single figure with 4 figures
- [ ] Update part5.tex: fix all text references and subplot letters
- [ ] Update appendix.tex: fix all references and subplot letters
- [ ] Verify: compile LaTeX and check all cross-references resolve
- [ ] Verify: figure numbering is sequential and consistent
