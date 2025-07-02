import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./logdir'),
    record_shapes=True,
    with_stack=True
) as prof:
    for step in range(5):
        with record_function("my_op"):
            x = torch.randn(10, 10, device='cuda')
            y = torch.matmul(x, x)
        prof.step()