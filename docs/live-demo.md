# Live demo

A real [`simplesvgd.update()`][simplesvgd.update] run, recorded with
[`RerunConfig`][simplesvgd.RerunConfig] and embedded below via the
[Rerun](https://rerun.io) web viewer -- no install required, this is the
same viewer you'd get locally with `RerunConfig(enabled=True)`.

300 particles start from a diffuse Gaussian and are pushed by SVGD towards
the four modes of the Himmelblau function (the same test function used in
the [mini-tutorial](index.md#defining-an-example-target)). Drag the
timeline at the bottom to scrub through the 300 iterations; the particle
cloud and the two diagnostic traces (`particle_variance`,
`repulsion_ratio`) stay in sync.

<iframe
    src="https://app.rerun.io/version/0.37.1?url=https://larsgeb.github.io/simpleSVGD/assets/rerun/himmelblau_demo.rrd"
    style="width: 100%; height: 700px; border: none;"
    allowfullscreen>
</iframe>

If the embed above doesn't load (e.g. an ad blocker or a strict browser
privacy mode blocking the `app.rerun.io` iframe), download the recording
and open it locally instead:

```sh
pip install simplesvgd[rerun]
rerun --web-viewer "https://larsgeb.github.io/simpleSVGD/assets/rerun/himmelblau_demo.rrd"
```

The script that generated this recording is at
[`docs/assets/rerun/generate_demo.py`](https://github.com/larsgeb/simpleSVGD/blob/master/docs/assets/rerun/generate_demo.py).
