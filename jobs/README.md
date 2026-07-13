# Running jobs

**Never run these unbounded on a shared machine.** A Mamba-2 Jacobian exhausted the
Spark's 119GB of *unified* memory and wedged the host — kernel alive, userspace dead.
On unified memory a GPU OOM is a system OOM: it does not raise `CUDA out of memory`,
it kills the box, and it takes every other user's job with it.

```bash
sbatch jobs/jlens.sbatch experiments/published_sweep.py --models 8B --purge
squeue; scancel <jobid>
```

`--mem=80G` is a hard cgroup limit. If a job exceeds it, Slurm kills **the job**.

## Memory notes per model class

| model class | why it is expensive |
|---|---|
| dense transformer | Jacobian = one backward per `dim_batch` chunk of `d_model`. Scales with `d_model²` for the accumulator, plus activations. |
| **Mamba / SSM hybrid** | **The recurrent scan's autograd graph is unrolled over the sequence.** Far worse than attention for a Jacobian. Start at `dim_batch=8`, `max_seq_len=64`, and watch RSS before scaling up. This is what took the box down. |
| MoE | all experts materialise in bf16 regardless of active params (`Nemotron-3-Nano-30B-A3B` is ~60GB, not 6GB). |

Until the Spark is in the Slurm cluster (`~/Workspace/slurm-add-spark.sh`), cap manually:

```bash
systemd-run --user --scope -p MemoryMax=80G -p MemorySwapMax=0 \
  .venv/bin/python experiments/whatever.py
```
