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

Generate plots from profiler logs

```bash
python jaxprof.py --help
```
