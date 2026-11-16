# Force2Act

Force2Act is a policy for contact-rich manipulation. A **global force policy** plans chunks of poses and
wrenches with flow matching, seeing force relative to its recent magnitude. A **Mixture of Force Experts
(MoFE)** sets the normal force at every control step from seven learned contact laws (pushing, two sticking
laws, three sliding levels, insertion). A router chooses among the laws that the measured contact geometry
admits. A hybrid motion–force controller executes the result.

This repository contains Force2Act (training and inference) and **Force2Act-Bench**, five contact-rich
manipulation tasks in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis).

| | |
|---|---|
| Models | [`k3rth1v5n/Force2Act-Sim`](https://huggingface.co/k3rth1v5n/Force2Act-Sim) |
| Demonstrations | [`k3rth1v5n/Force2Act-Bench-Data`](https://huggingface.co/datasets/k3rth1v5n/Force2Act-Bench-Data) |

## Installation

```bash
conda create -n force2act python=3.11 && conda activate force2act
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[bench,train]"
```

The benchmark renders its cameras with EGL and needs an NVIDIA GPU.

## Evaluate Force2Act

```bash
force2act-bench --task pivot_block
```

This downloads the trained policy and MoFE, runs the task's full protocol in parallel (`--workers`, default
8) and writes `results/pivot_block/summary.md` and `results.jsonl`. Add `--video` to save one video per
episode, or `--trials N` for a quick run.

| Task | Conditions | Episodes | Force2Act |
|---|---|---|---|
| `pivot_block` | block mass 0.4–0.9 kg | 6 × 25 | 93.3% |
| `push_box` | incline 0°, 2°, 4° | 3 × 10 | 100% |
| `bottle_wiping` | random ink line | 50 | 98.0% |
| `plug_insertion_round` | hole at the demonstrated position, displaced by 2 mm | 2 × 25 | 96.0% |
| `plug_insertion_usb` | socket at the demonstrated position | 25 | 76.0% |

## Evaluate your own policy

A policy is any object with `reset(seed)`, `act(observation) -> Action` and `needs_image()`:

```python
from force2act_bench.sim import Action

class MyPolicy:
    def reset(self, seed): ...
    def needs_image(self): return True
    def act(self, observation):
        # observation: tcp_position, tcp_quaternion_wxyz (world), tcp_wrench (environment on tool,
        # TCP axes), joint_positions, rgb {camera: H x W x 3 uint8}
        return Action(tcp_position=..., tcp_quaternion_wxyz=...)

def make_policy(**kwargs):
    return MyPolicy()
```

```bash
force2act-bench --task push_box --policy my_package.my_policy:make_policy --control impedance
```

| `--control` | The action sets |
|---|---|
| `position` | the TCP pose, tracked by inverse kinematics and joint PD control |
| `impedance` | the TCP pose, tracked by a Cartesian impedance (2000 N/m, 100 Nm/rad) |
| `hybrid` | the TCP pose, and the wrench on the axes in `action.force_control_axes`, expressed in the frame `action.force_frame_rotation_wxyz` |

Options can also be collected in a YAML file and passed with `--config` (see `configs/bench/`).

## Train

```bash
hf download k3rth1v5n/Force2Act-Bench-Data --repo-type dataset --local-dir data

# global force policy, one per task (60 epochs)
force2act-train --config configs/policy/pivot_block.yaml --output runs/pivot_block

# MoFE, one model for all tasks (10 epochs)
force2act-train-mofe --configs configs/policy/*.yaml --output runs/mofe

force2act-bench --task pivot_block --checkpoint runs/pivot_block --mofe runs/mofe
```

The MoFE ablations of the paper are available as flags: `--no-gates`, `--router-sees-direction`,
`--one-per-case`, `--offset-bound 60` and `--lam 0`.

## Use Force2Act in your own loop

```python
from force2act.agent import Force2ActAgent

agent = Force2ActAgent.from_pretrained("hf://k3rth1v5n/Force2Act-Sim/pivot_block")
agent.reset()
command = agent.act(position, quaternion_wxyz, wrench_world, image)   # at 15 Hz
# command.position, command.quaternion_wxyz: pose target
# command.force_axes, command.frame_quaternion_wxyz, command.wrench: hybrid force command
```

## Citation

```bibtex
@inproceedings{force2act2027,
  title     = {Force2Act},
  author    = {},
  booktitle = {},
  year      = {2027}
}
```

## Acknowledgements

The Force2Act-Bench simulation builds on the Genesis contact simulator by Shuo Li. The normalizer and EMA utilities are adapted from [ManiFlow](https://github.com/geyan21/ManiFlow_Policy) (MIT).
The action head follows [DiT](https://github.com/facebookresearch/DiT), the flow-time distribution follows
[π0](https://github.com/Physical-Intelligence/openpi), and inference uses
[real-time chunking](https://github.com/Physical-Intelligence/real-time-chunking-kinetix). Demonstrations use the
[LeRobot](https://github.com/huggingface/lerobot) format.

## License

MIT
