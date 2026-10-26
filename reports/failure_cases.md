# Failure-case analysis

What goes wrong, how often, and what it looks like (T37). Produced by `python run.py failures`: B1, B2, M1 and M2 each sample the same 100 test rooms 64 times (6,400 samples per method; frozen configuration, seed 0). M1 and M2 start from the same draws, so sample *i* of M2 is sample *i* of M1 after latent optimization.

Every sample is classified by the hard checks it breaks: **out of room** (H1), **overlap** (H2), **door blocked** (H3), **unreachable** (H4). A valid sample is **poor** if its rule score is below 0.5, otherwise **good**.

Tables: `reports/tables/failure_causes.csv` and `failure_cases.csv`. Gallery: `reports/figures/failure_cases.png`.

## 1. How often each failure occurs

Share of all samples of a method. A sample can break several checks.

| | B1 | B2 | M1 | M2 |
|---|---|---|---|---|
| **Valid** | 9.9% | 58.6% | 14.4% | **67.8%** |
| of which good (score at least 0.5) | 0.2% | 37.9% | 11.2% | 50.1% |
| of which poor | 9.7% | 20.8% | 3.1% | 17.6% |
| Breaks H1, out of room | 0.0% | 0.3% | 29.7% | 4.0% |
| Breaks H2, overlap | 70.9% | 1.2% | 56.7% | 6.4% |
| Breaks H3, door blocked | 40.9% | 31.9% | 34.3% | 9.6% |
| Breaks H4, unreachable | 63.3% | 34.9% | 45.0% | 28.4% |
| Breaks several checks | 59.8% | 26.5% | 53.0% | 10.8% |
| Breaks only H4 | 7.1% | 8.5% | 1.9% | **18.5%** |
| Rooms with no valid layout at all | 16% | 0% | 11% | 0% |

## 2. What the numbers say

**Latent optimization repairs what it targets.** From M1 to M2, on the same candidates:

| Check | M1 | M2 | Removed |
|---|---|---|---|
| Out of room | 29.7% | 4.0% | 87% |
| Overlap | 56.7% | 6.4% | 89% |
| Door blocked | 34.3% | 9.6% | 72% |
| Unreachable | 45.0% | 28.4% | 37% |

The first three have a term in the optimization loss. Reachability has none, because it is not differentiable; it improves only as a side effect of items moving apart.

**Reachability is M2's main remaining failure.** 28.4% of M2's samples leave an item unreachable, which is 88% of its invalid samples, and 18.5% of all samples fail on reachability *alone*: nothing overlaps, nothing leaves the room, the door is free, but there is no 0.6 m path to some item's front. That is more than half of M2's failures (case 4 in the gallery). Any improvement of M2 has to start here.

**The raw CVAE mostly fails on overlap and on the walls.** M1's decoder places each item about 0.2 m from where a good layout would have it. That is enough for two neighbouring items to overlap (56.7%) and for a wall item to stick through its wall (29.7%). The Sigmoid keeps an item's *centre* inside the room, not its footprint. These are exactly the small violations that a few gradient steps remove (cases 7 and 8).

**B2 fails differently.** It almost never overlaps (1.2%), because it redraws an item that would. It fails on the door zone (31.9%) and on reachability (34.9%), which it never looks at.

**Valid is not the same as good.** A quarter of M2's valid samples score below 0.5 (17.6% of 67.8%), and a third of B2's (20.8% of 58.6%). The typical poor layout has an item that belongs against a wall standing in the open (cases 6 and 11): it breaks no hard rule and looks wrong. This is what the evaluator's ranking is for: the top 3 shown average 0.85, well above the 0.66 of all valid samples.

**No collapse.** No room of M2 has near-identical valid layouts (the least varied room still averages 1.05 m between them), and every room gets valid layouts. M1 leaves 11% of rooms without any valid layout, B1 16%.

## 3. Gallery

Red edges mark the items that break a check; the label under the item's name says which. The shaded floor is what can be walked to from the door.

![Failure cases](figures/failure_cases.png)

| Case | Method | What happened | Cause |
|---|---|---|---|
| 1 | M2 | The TV unit sticks through the east wall by a few centimeters | Out of room: a small protrusion that the optimization left |
| 2 | M2 | The side table overlaps the sofa's corner | Overlap. A small item next to a large one: the residual overlap is above the 0.005 m² tolerance |
| 3 | M2 | The armchair stands in the door's clearance zone | Door blocked |
| 4 | M2 | Nothing overlaps, yet the sofa, the armchair and the side table cannot be reached: the walkable floor stops short of them | Unreachable only: the passages left between the items and the walls are narrower than the 0.6 m a person needs. The loss has no term for this |
| 5 | M2 | Two pairs of items still overlap, and the sofa and the side table are cut off | Several checks at once: a candidate far from any valid layout |
| 6 | M2 | Valid, score 0.23: the TV unit stands away from its wall, less than a meter in front of the sofa | Poor. No hard rule is broken; alignment and the sofa-to-TV distance are |
| 7, 8 | M1, then M2 | Before: the sofa and the TV unit stick out of the room and the armchair blocks the door. After: valid, score 0.86 | A successful repair of the same candidate |
| 9, 10 | M1, then M2 | Before: overlap, door blocked, unreachable. After: the overlap is gone, but the bookshelf still stands in the door zone and the sofa is still cut off | A failed repair. Probably the bookshelf would have to move too far: the anchor to the starting point holds every candidate back |
| 11 | B2 | Valid, score 0.23: the sofa faces the door wall, and the TV unit stands on the wall to its side | Poor. Each position was copied from a different training layout; each is plausible alone, together they are not |
| 12 | B1 | The sofa lies on top of the TV unit; the armchair blocks the door | Random placement |

## 4. What follows

1. **A reachability term is the most valuable next step for M2.** More than half of its failures break nothing else. Options: a differentiable stand-in (for example a penalty on gaps narrower than 0.6 m in front of each item), or the CNN evaluator as a surrogate, with a test for the optimizer exploiting it.
2. **The door zone is repaired least of the three targeted checks** (72%). An item inside the zone has to travel up to 0.9 m, which the anchor resists. A larger door weight or more steps for such candidates would be the first thing to try.
3. **Ranking matters as much as validity** for what the user sees, because a quarter of the valid layouts are poor.
4. **The checker must stay the final authority.** Every one of these failures was caught by it, and none would have been caught reliably by the CNN alone at the tolerance boundary.
