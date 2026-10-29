# One task boundary in parameter space

One figure. A task boundary is a **fork**, and this is what the fork looks like
when both of its branches are recorded and measured.

```bash
python reports/final/landscape/run_trajectory.py --task 5 --phase local  --stride 10
python reports/final/landscape/run_trajectory.py --task 5 --phase global --stride 5
python reports/final/landscape/project.py --task 5 --n 31      # ~6 min
python reports/final/landscape/make_figures.py                 # seconds
```

---

## What a task boundary is

When task 5 arrives, the global policy carried in from task 4 is used twice.

1. It is **cloned** and the clone is trained on task 5 alone for 1000
   iterations. That clone becomes the **specialist**, whose value is the floor
   the constraint enforces.
2. The original is **consolidated** for 505 iterations under that floor, and
   becomes the global policy carried on to task 6.

`local_init_fresh` is false in this configuration, so the specialist really is
cloned from the incoming global rather than initialised afresh. The two phases
therefore leave from the same point, which is what makes it legitimate to draw
them on one plane.

## Four figures

`task_boundary_combined` is the overview: six panels, columns the three acts and
rows the two renderings, so each column is the same question answered flat and
as a surface. Panels (a) to (f), one legend, one colourbar, one key. It carries
no data the other three do not; it is them laid out together.

Two layout facts worth knowing. The surface panels keep the box, camera and
crop their standalone figures give them, so they are rendered at full size
rather than squeezed into a smaller cell; the flat panels are the ones narrowed
to fit, which is what keeps the figure wider than it is tall. The flat row
shares one crop and therefore labels PC 1 and PC 2 once.

**The surface row carries no text of its own.** No axis titles, no tick
labels, no A, B or C letters. It sits under a flat row that shares its crop,
its colourbar and its key, so anything written on it would say a second time
what is already said once, in three cells too small to take it. Only the
panel letters (d), (e) and (f) remain, above the scenes rather than in them.
Two things follow from the labels going. The grid darkens a step, to
`#CED4DA`, since there is nothing else left to frame the box; and the cameras
pull in to 0.90, because the cells were sized for labels that are no longer
there.

The other three stand alone and are described below.

## Three standalone figures, in the order the phases happen

`task_boundary_2d` is the one to read first. Three flat maps of the same plane.

1. **Specialist training, on tasks 1 to 4.** A sits on high ground, the branch
   descends into the low ground, B ends there. **No constraint line**, because
   at this point in the sequence there is none: the floor is the specialist's
   own value, which does not exist until B does.
2. **The specialist sets the floor, on task 5.** Where the constraint comes
   from. A sits where task 5 is lost, B where it is held, and the green line is
   the edge between them.
3. **Consolidation, on all five tasks seen.** The branch runs from A to C and
   stays on the admissible side. Read on all five rather than on tasks 1 to 4,
   because that is the field on which A and C are comparable.

`specialist_phase_3d` and `consolidation_phase_3d` are the same data as
surfaces, one phase each, zoomed to the branch. The specialist figure puts the
same path on two surfaces: it falls on tasks 1 to 4 and climbs the task-5 cliff.
The consolidation figure drains the colour from the ground the constraint rules
out, so the admissible region reads as an area rather than as a line, and lays
the constraint edge on the surface.

**Both consolidation views are read on the mean over all five tasks seen**,
not on tasks 1 to 4. A has not learned task 5 yet, so judging both on the
earlier four flatters A: it reads 0.71 against C's 0.42 and consolidation looks
like a loss. On all five it is 0.56 against 0.53, which is the comparison that
is actually fair. Panels 1 and 2 keep their own single fields, because the
specialist's trade-off is exactly between them.

## What the figures show

| | position | tasks 1-4 | task 5 |
|---|---|---:|---:|
| **A** global in, from task 4 | (0, 0) | **0.71** | −0.06 |
| **B** specialist out, sets the floor | (−11.5, −12.6) | **0.02** | **1.00** |
| **C** global out, to task 6 | (−13.7, +9.0) | **0.42** | **1.00** |

Both branches leave A and fork to opposite sides. Both end with task 5 solved
at 1.00. They end **0.40 apart on tasks 1 to 4**. The specialist reaches a
solved new task by giving the earlier ones up; consolidation reaches the same
new-task score and keeps a large part of them.

## On the constraint looking loose

A reviewer will ask why the admissible region is half the plane. Because task 5
is near-binary over it, not because the budget is slack: **50%** of the plane
scores above 0.9 on task 5 and **38%** below 0.1, with only **12%** in between.
Loosening the budget from 0.05 to 0.20 widens the admissible region by seven
points of area. The line is set by the cliff, not by $\epsilon$, and panel 2
shows it sitting on that cliff.

## How it is drawn, and what was discarded

Each decision was made by building the alternative and comparing.

**One scalar per panel.** Encoding the objective as height and the constraint
as colour of one surface was tried first and discarded: a reader cannot then
say what the landscape is a landscape of.

**The colour scale runs blue through violet to red, with no pale band.** A
near-white low end is indistinguishable from the page, and a diverging scale
with a pale centre implies a neutral value that a normalized score does not
have, which put C on the "bad" side of a midpoint that means nothing.

**The constraint line is green.** Red on the red half of a diverging scale is
not legible.

**The constraint is absent from panel 1 on purpose.** Drawing it during
specialist training would be anachronistic, since the specialist defines it.

**Arrows sit mid-branch,** because at the end the head lands under the
endpoint marker and is never seen.

**Nothing is smoothed except the constraint curve.** The fill is the raw 61 by
61 measurement. Without a light blur on the shortfall field the curve breaks
into noise islands that read as structure.

**A parameter-space ball was drawn and then removed** at the author's request.
The measurement behind it is kept below, because reviewers ask the question.

**The two regions are named in a legend, not in the caption.** A caption that
has to explain what a colour means is doing the legend's job.

The rules this figure set produced are written up for reuse in
`~/.claude/skills/algorithmic-art-academic/SKILL.md`, section F.5c.

## Caveats, stated plainly

**The run.** `results/duel_v2_matched_seed0` is the only Setting B run that
kept per-task checkpoints. It matches the reported runs on every architectural
and environment field (`mlp_ac_sep`, hidden `[512, 512, 256]`,
`task_conditioned`, `goal_in_obs`, `view_max 5`, `max_steps 160`, the frozen
fifty-task dataset, $\epsilon = 0.04$, $\eta = 1.0$, $\mu_{\max} = 10$) but is a
different run instance from the three seeds behind Table 2, and it was stopped
during task 6.

**Both phases are re-executions.** PPO is stochastic and the rollout seeds
differ, so these branches are not the exact ones the original run took. The
original run's own `specialist` and `consolidated` checkpoints sit 20.9 and
16.6 off this plane out of total distances of 21.4 and 17.8, which is why they
are reported in `project.py` output but **not drawn**. A point that is 95% off
a plane must not be placed on it. The structure of the fork is the finding, not
the exact endpoint.

**Resolution and noise.** 31 by 31 grid, twelve greedy episodes per task per
point, one evaluation seed. Repeat measurements of the same quantity move by
about 0.06, so the 0.38 gap between the two branch endpoints is well outside
noise and the surface's small bumps are inside it.

**Rendering.** PC1 and PC2 are both distances in parameter space, so the panel
is drawn at equal aspect and the canvas is sized to the data's own ratio. The
fill is box blurred over a 3 by 3 window for the contour geometry only; every
number quoted in the figure and in this file is unsmoothed.

## A note on the SVGs, for slides

Plotly renders a 3D scene with WebGL, and WebGL has no vector form. Every
surface panel is therefore embedded in the SVG as a **bitmap**, one per scene,
while everything else is real vector. Opened in PowerPoint or Illustrator:

| file | 3D bitmaps | vector |
|---|---|---|
| `task_boundary_2d` | none | fully editable |
| `specialist_phase_3d` | 2 | axes and labels only |
| `consolidation_phase_3d` | 1 | axes and labels only |
| `task_boundary_combined` | 3 | the whole flat row, plus all text |

So in the combined figure the three surface panels arrive as three image
objects that cannot be broken down further. That is the library, not the
figure, and no export setting changes it. `SVG_SCALE` controls how sharp those
bitmaps are; it is set to 2, which Plotly oversamples to roughly 4x the panel
size, enough for print without the file becoming megabytes of base64.

**Two bare versions exist for slides.** `--bare` rebuilds the two surface
figures with no text in them at all: no axis titles, no tick labels, no A, B
or C letters, no heading, no colourbar and no key. Just the surfaces, the
branches, the markers and the grid, with the grid a step darker so it still
reads once the labels that framed it are gone, and the camera pulled in so the
art fills the canvas rather than carrying a white border into the slide.

```bash
python reports/final/landscape/make_figures.py          # the four framed figures
python reports/final/landscape/make_figures.py --bare   # *_3d_bare, for slides
```

`--bare` also builds `specialist_task5_3d_bare`, which the framed set has no
equivalent of: panel (e) alone, the task-5 cliff on its own canvas and at a
size worth projecting, since it is the surface the constraint is cut from.

They are the same scenes as the framed versions, so the surfaces still arrive
as bitmaps and the labelling is meant to be added on the slide. With the text
gone these files hold nothing but their bitmaps: one `<g>`, one `<image>` per
scene, and no vector at all. The SVG is then only worth taking over the PNG
for keeping the panels as separate objects.

**Every SVG is swept before it is written.** Plotly emits a `<g>` for each of
the twenty-odd trace types it knows whether the figure uses one or not, a clip
rectangle for each of those empty layers, invisible hit targets behind legend
entries and colourbars, and a rectangle of solid paper behind everything. None
carries any of the figure, and in PowerPoint each is a selectable object: the
paper rectangle covers the whole canvas and is invisible against a white
slide, so it is found by accident. `clean_svg` removes them and reports what
it took. It removes a white rectangle only when that rectangle is the full
canvas at full opacity, because the pads behind the A, B and C labels are also
white rectangles, at 88%, and the letters stop being readable without them.

A pure-vector rebuild of the combined figure in matplotlib was tried and
dropped. Its mplot3d emits real paths, so the SVG held no bitmaps, but
mplot3d flat-shades one polygon per cell where Plotly interpolates colour
across each quad and lights the result, and no vector format can carry either:
SVG 1.1 has linear and radial gradients only, no mesh gradient and no Gouraud.
Every vector route lands on flat polygons. The surfaces read visibly plainer
than these, which was not worth the editability.
3. **Consolidation, on all five tasks seen.** The branch runs from A to C and
   stays on the admissible side. Read on all five rather than on tasks 1 to 4,
   because that is the field on which A and C are comparable.

`specialist_phase_3d` and `consolidation_phase_3d` are the same data as
surfaces, one phase each, zoomed to the branch. The specialist figure puts the
same path on two surfaces: it falls on tasks 1 to 4 and climbs the task-5 cliff.
The consolidation figure drains the colour from the ground the constraint rules
out, so the admissible region reads as an area rather than as a line, and lays
the constraint edge on the surface.

**Both consolidation views are read on the mean over all five tasks seen**,
not on tasks 1 to 4. A has not learned task 5 yet, so judging both on the
earlier four flatters A: it reads 0.71 against C's 0.42 and consolidation looks
like a loss. On all five it is 0.56 against 0.53, which is the comparison that
is actually fair. Panels 1 and 2 keep their own single fields, because the
specialist's trade-off is exactly between them.

## What the figures show

| | position | tasks 1-4 | task 5 |
|---|---|---:|---:|
| **A** global in, from task 4 | (0, 0) | **0.71** | −0.06 |
| **B** specialist out, sets the floor | (−11.5, −12.6) | **0.02** | **1.00** |
| **C** global out, to task 6 | (−13.7, +9.0) | **0.42** | **1.00** |

Both branches leave A and fork to opposite sides. Both end with task 5 solved
at 1.00. They end **0.40 apart on tasks 1 to 4**. The specialist reaches a
solved new task by giving the earlier ones up; consolidation reaches the same
new-task score and keeps a large part of them.

## On the constraint looking loose

A reviewer will ask why the admissible region is half the plane. Because task 5
is near-binary over it, not because the budget is slack: **50%** of the plane
scores above 0.9 on task 5 and **38%** below 0.1, with only **12%** in between.
Loosening the budget from 0.05 to 0.20 widens the admissible region by seven
points of area. The line is set by the cliff, not by $\epsilon$, and panel 2
shows it sitting on that cliff.

## How it is drawn, and what was discarded

Each decision was made by building the alternative and comparing.

**One scalar per panel.** Encoding the objective as height and the constraint
as colour of one surface was tried first and discarded: a reader cannot then
say what the landscape is a landscape of.

**The colour scale runs blue through violet to red, with no pale band.** A
near-white low end is indistinguishable from the page, and a diverging scale
with a pale centre implies a neutral value that a normalized score does not
have, which put C on the "bad" side of a midpoint that means nothing.

**The constraint line is green.** Red on the red half of a diverging scale is
not legible.

**The constraint is absent from panel 1 on purpose.** Drawing it during
specialist training would be anachronistic, since the specialist defines it.

**Arrows sit mid-branch,** because at the end the head lands under the
endpoint marker and is never seen.

**Nothing is smoothed except the constraint curve.** The fill is the raw 61 by
61 measurement. Without a light blur on the shortfall field the curve breaks
into noise islands that read as structure.

**A parameter-space ball was drawn and then removed** at the author's request.
The measurement behind it is kept below, because reviewers ask the question.

**The two regions are named in a legend, not in the caption.** A caption that
has to explain what a colour means is doing the legend's job.

The rules this figure set produced are written up for reuse in
`~/.claude/skills/algorithmic-art-academic/SKILL.md`, section F.5c.

## Caveats, stated plainly

**The run.** `results/duel_v2_matched_seed0` is the only Setting B run that
kept per-task checkpoints. It matches the reported runs on every architectural
and environment field (`mlp_ac_sep`, hidden `[512, 512, 256]`,
`task_conditioned`, `goal_in_obs`, `view_max 5`, `max_steps 160`, the frozen
fifty-task dataset, $\epsilon = 0.04$, $\eta = 1.0$, $\mu_{\max} = 10$) but is a
different run instance from the three seeds behind Table 2, and it was stopped
during task 6.

**Both phases are re-executions.** PPO is stochastic and the rollout seeds
differ, so these branches are not the exact ones the original run took. The
original run's own `specialist` and `consolidated` checkpoints sit 20.9 and
16.6 off this plane out of total distances of 21.4 and 17.8, which is why they
are reported in `project.py` output but **not drawn**. A point that is 95% off
a plane must not be placed on it. The structure of the fork is the finding, not
the exact endpoint.

**Resolution and noise.** 31 by 31 grid, twelve greedy episodes per task per
point, one evaluation seed. Repeat measurements of the same quantity move by
about 0.06, so the 0.38 gap between the two branch endpoints is well outside
noise and the surface's small bumps are inside it.

**Rendering.** PC1 and PC2 are both distances in parameter space, so the panel
is drawn at equal aspect and the canvas is sized to the data's own ratio. The
fill is box blurred over a 3 by 3 window for the contour geometry only; every
number quoted in the figure and in this file is unsmoothed.

## A note on the SVGs, for slides

Plotly renders a 3D scene with WebGL, and WebGL has no vector form. Every
surface panel is therefore embedded in the SVG as a **bitmap**, one per scene,
while everything else is real vector. Opened in PowerPoint or Illustrator:

| file | 3D bitmaps | vector |
|---|---|---|
| `task_boundary_2d` | none | fully editable |
| `specialist_phase_3d` | 2 | axes and labels only |
| `consolidation_phase_3d` | 1 | axes and labels only |
| `task_boundary_combined` | 3 | the whole flat row, plus all text |

So in the combined figure the three surface panels arrive as three image
objects that cannot be broken down further. That is the library, not the
figure, and no export setting changes it. `SVG_SCALE` controls how sharp those
bitmaps are; it is set to 2, which Plotly oversamples to roughly 4x the panel
size, enough for print without the file becoming megabytes of base64.

**`task_boundary_combined_vector` is the answer to this.** It redraws the same
six panels with matplotlib, whose mplot3d emits real paths, so the SVG contains
no bitmaps at all: 5170 vector paths, 147 editable text elements and 643 named
groups. Text is written as text rather than outlines (`svg.fonttype = "none"`),
and the colourbar is built from stacked rectangles because matplotlib would
otherwise emit its solids as an image. Every axis, tick, label, contour band
and surface quad can be selected and edited.

```bash
python reports/final/landscape/make_figures_vector.py   # svg, pdf and a png preview
```

Two rendering details decide whether it looks like the Plotly version or like
a mosaic, and they pull against each other.

`plot_surface` draws one flat polygon per grid cell. A vector renderer
antialiases each polygon against the page rather than against its neighbour,
so at zero line width a hairline of white is left on every shared edge and a
mesh lies over the surface. A thin edge on each quad closes it. But a quad
here is only about 1.3 pt across, so an edge of 0.35 pt covers a quarter of
one, bleeds each cell's colour into its neighbours, and washes the relief out
into smooth bands: the seams go and the landscape goes with them. 0.05 pt is
the width that closes the seams and keeps the detail.

That edge must be left to `plot_surface`, which already passes the face
colours as edge colours in the order its own depth sort will use. Assigning
them afterwards from `get_facecolor` hands over the same array in a different
order and speckles the surface red and blue.

Neither effect is visible in a PNG preview, because Agg antialiases polygons
against each other and matplotlib's own rasteriser is not what opens the file.
Both were judged by rendering the PDF through cairo at 260 to 420 dpi, which
is what a vector renderer does with it.

Lighting is off on purpose. Plotly's surfaces have it and it would add relief,
but matplotlib rescales face brightness over 0.3 to 1.0 to apply it, after
which a colour no longer means the value the colourbar maps it to. The relief
survives without it once the edge width is right.

The display grid stays at the measured 61 by 61, which is 3600 quads a
surface, fine enough once the seams are closed and still few enough to open in
an editor. Upsampling it was tried and reverted: it multiplies the path count
by the square of the factor and buys nothing the edge width does not.

It is the same measured grid and the same palette as the Plotly version, drawn
by a second renderer. Two differences follow from mplot3d rather than from the
data. It depth-sorts polygons but treats a 3D line as one artist, so a branch
is drawn wholly in front of its surface, which is what is wanted here since the
branch is the thing to follow. And the two surfaces in panel (f) tile the plane
rather than intersecting, so the painter's algorithm has nothing to get wrong
between them.

---

## Conventions

The academic template from `report/acviz.py`. White ground, a sequential scale
built from the palette rather than a rainbow, blue for ours and amber for the
specialist as everywhere else in the paper, mono numerals, PNG at 300 dpi and
SVG from the same figure object. Set
`FIGS_FOR_PAPER=1` to drop the footnote, which a LaTeX caption would otherwise
duplicate.
