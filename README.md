# jax-profiler

Memory profiler for JAX

## Usage

Create profiler logs

```python
from jaxprof import JaxProfiler

profiler = JaxProfiler()
def some_jax_code():
    ...
    profiler.capture()
    ...
```

or run it in the background

```python
profiler.capture_in_background()
```

Generate plots from profiler logs

```bash
python jaxprof.py --help
```
