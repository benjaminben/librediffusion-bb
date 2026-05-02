# test_engine.py fixtures

Three fixtures live here. All are 1024×1024 RGBA. `test_engine.py` resizes
them to the engine's resolution at load time, so a single fixture set
works against any output resolution.

| File | Purpose |
|---|---|
| `source.png` | source image for img2img |
| `canny.png` | canny edges of `source.png` — the **in-domain** control signal |
| `canny_of_different.png` | canny of an **unrelated** image — the **out-of-domain** test that verifies ControlNet is actually consuming the control input rather than ignoring it |

## Acceptance criterion

When the same `source.png` is run with `--control canny.png` vs
`--control canny_of_different.png`, the output images must visibly differ
and the second output should follow the canny structure rather than the
source structure. If they look identical, the engine is not consuming
its control input.

## Regenerating fixtures

If the committed fixtures are missing or stale, run:

    python tools/test_data/_generate_fixtures.py

That script uses Pillow + an unrelated public-domain photo to produce all
three PNGs deterministically.
