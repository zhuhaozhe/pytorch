import torch
linear = torch.nn.Linear(128, 128)
compiled = torch.compile(linear)
x = torch.rand(10, 128)
with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    compiled(x)
    compiled(x)

from torch._dynamo import config
config.error_on_recompile = True
from torch.utils import ThroughputBenchmark
with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    bench = ThroughputBenchmark(compiled)
    bench.add_input(x)
    stats = bench.benchmark(
        num_calling_threads=10,
        num_warmup_iters=100,
        num_iters=100,
    )
    print(stats)