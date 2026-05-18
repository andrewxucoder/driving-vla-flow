from __future__ import annotations

import argparse
import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import torch
import yaml
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from driving_vla.data.toy_dataset import ToyDrivingConfig, ToyDrivingDataset
from driving_vla.evaluation.reward import RewardWeights, rank_trajectories
from driving_vla.models.flow_matching import ConditionalFlowMatcher, sample_flow
from driving_vla.models.regression_head import WaypointRegressionHead
from driving_vla.models.vla_encoder import DrivingVLAEncoder


MODEL_CHOICES = ("flow", "regression", "diffusion", "all")
DEFAULT_CHECKPOINTS = {
    "flow": Path("outputs/toy_flow/model.pt"),
    "regression": Path("outputs/regression/model.pt"),
    "diffusion": Path("outputs/diffusion/model.pt"),
}
MODEL_STYLES = {
    "flow": {"label": "Flow", "color": "tab:blue", "marker": "x", "linestyle": "--"},
    "regression": {"label": "Regression", "color": "tab:orange", "marker": "s", "linestyle": "-."},
    "diffusion": {"label": "Diffusion", "color": "tab:purple", "marker": "^", "linestyle": ":"},
}


@dataclass
class ModelRunner:
    name: str
    label: str
    color: str
    marker: str
    linestyle: str
    cfg: dict
    encoder: DrivingVLAEncoder
    generator: nn.Module
    predict_fn: Callable[[nn.Module, torch.Tensor, dict], torch.Tensor]

    def predict(self, state: torch.Tensor, cmd_id: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cond = self.encoder(state, cmd_id)
            return self.predict_fn(self.generator, cond, self.cfg)

    def predict_candidates(self, state: torch.Tensor, cmd_id: torch.Tensor, num_samples: int) -> torch.Tensor:
        with torch.no_grad():
            cond = self.encoder(state, cmd_id).repeat(num_samples, 1)
            return self.predict_fn(self.generator, cond, self.cfg)


def device_from_cfg(value: str) -> torch.device:
    if value == "auto":
        value = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    return torch.device(value)


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"未找到模型文件：{path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def build_dataset(cfg: dict, seed: int) -> ToyDrivingDataset:
    data_section = cfg["data"]
    data_cfg = ToyDrivingConfig(
        num_samples=data_section["num_samples"],
        horizon=data_section["horizon"],
        state_dim=data_section["state_dim"],
        language_vocab=data_section.get("language_vocab"),
    )
    return ToyDrivingDataset(data_cfg, seed=seed)


def build_encoder(cfg: dict, device: torch.device) -> DrivingVLAEncoder:
    data_section = cfg["data"]
    return DrivingVLAEncoder(
        state_dim=data_section["state_dim"],
        num_commands=len(data_section["language_vocab"]),
        text_embed_dim=cfg["model"]["text_embed_dim"],
        latent_dim=cfg["model"]["latent_dim"],
    ).to(device)


def build_flow_model(cfg: dict, checkpoint: dict, device: torch.device) -> tuple[nn.Module, Callable[[nn.Module, torch.Tensor, dict], torch.Tensor]]:
    data_section = cfg["data"]
    model = ConditionalFlowMatcher(
        horizon=data_section["horizon"],
        traj_dim=2,
        cond_dim=cfg["model"]["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["flow"])

    def predict_fn(generator: nn.Module, cond: torch.Tensor, model_cfg: dict) -> torch.Tensor:
        return sample_flow(generator, cond, steps=model_cfg["model"]["flow_steps"])

    return model, predict_fn


def build_regression_model(cfg: dict, checkpoint: dict, device: torch.device) -> tuple[nn.Module, Callable[[nn.Module, torch.Tensor, dict], torch.Tensor]]:
    data_section = cfg["data"]
    model = WaypointRegressionHead(
        horizon=data_section["horizon"],
        traj_dim=2,
        cond_dim=cfg["model"]["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["regression"])

    def predict_fn(generator: nn.Module, cond: torch.Tensor, model_cfg: dict) -> torch.Tensor:
        return generator(cond)

    return model, predict_fn


def build_diffusion_model(cfg: dict, checkpoint: dict, device: torch.device) -> tuple[nn.Module, Callable[[nn.Module, torch.Tensor, dict], torch.Tensor]]:
    try:
        module = importlib.import_module("driving_vla.models.diffusion_head")
    except ModuleNotFoundError as exc:
        raise RuntimeError("diffusion 适配器已预留，但当前项目尚未实现 driving_vla.models.diffusion_head") from exc

    model_cls = getattr(module, "TrajectoryDiffusionModel", None) or getattr(module, "ConditionalDiffusionModel", None)
    sample_diffusion = getattr(module, "sample_diffusion", None)
    if model_cls is None or sample_diffusion is None:
        raise RuntimeError("diffusion_head 需要提供 TrajectoryDiffusionModel/ConditionalDiffusionModel 和 sample_diffusion")

    data_section = cfg["data"]
    model = model_cls(
        horizon=data_section["horizon"],
        traj_dim=2,
        cond_dim=cfg["model"]["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["diffusion"])

    def predict_fn(generator: nn.Module, cond: torch.Tensor, model_cfg: dict) -> torch.Tensor:
        steps = model_cfg["model"].get("diffusion_steps", model_cfg["model"].get("flow_steps", 16))
        return sample_diffusion(generator, cond, steps=steps)

    return model, predict_fn


def build_runner(model_name: str, checkpoint_path: Path, base_cfg: dict, device: torch.device) -> ModelRunner:
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = checkpoint.get("cfg", base_cfg)
    encoder = build_encoder(cfg, device)
    encoder.load_state_dict(checkpoint["encoder"])
    builders = {
        "flow": build_flow_model,
        "regression": build_regression_model,
        "diffusion": build_diffusion_model,
    }
    generator, predict_fn = builders[model_name](cfg, checkpoint, device)
    encoder.eval(); generator.eval()
    style = MODEL_STYLES[model_name]
    return ModelRunner(
        name=model_name,
        label=style["label"],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        cfg=cfg,
        encoder=encoder,
        generator=generator,
        predict_fn=predict_fn,
    )


def find_sample_for_command(ds: ToyDrivingDataset, cmd_id: int) -> dict[str, torch.Tensor]:
    for sample in ds:
        if int(sample["cmd_id"].item()) == cmd_id:
            return sample
    raise ValueError(f"数据集中未找到 command id={cmd_id} 的样本")


def lane_center_for_command(cmd_id: int) -> float:
    if cmd_id == 1:
        return 3.5
    if cmd_id == 2:
        return -3.5
    return 0.0


def plot_trajectories(
    gt: np.ndarray,
    predictions: dict[str, np.ndarray],
    runners: list[ModelRunner],
    title: str,
    out_path: Path,
    flow_candidates: np.ndarray | None = None,
    best_flow_idx: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.plot(gt[:, 0], gt[:, 1], "o-", linewidth=2, markersize=3, color="tab:green", label="Ground truth")
    ax.scatter(gt[0, 0], gt[0, 1], c="tab:green", s=50, label="Start")
    ax.scatter(gt[-1, 0], gt[-1, 1], c="tab:red", s=50, label="GT end")

    if flow_candidates is not None:
        for idx, candidate in enumerate(flow_candidates):
            is_best = idx == best_flow_idx
            ax.plot(
                candidate[:, 0],
                candidate[:, 1],
                color="tab:blue" if is_best else "tab:cyan",
                linewidth=2.8 if is_best else 1.0,
                alpha=1.0 if is_best else 0.28,
                linestyle="-" if is_best else "--",
                label="Flow best" if is_best else ("Flow candidates" if idx == 0 else None),
            )
        if best_flow_idx is not None:
            best = flow_candidates[best_flow_idx]
            ax.scatter(best[-1, 0], best[-1, 1], c="tab:blue", s=60, label="Flow best end")

    for runner in runners:
        if runner.name == "flow" and flow_candidates is not None:
            continue
        pred = predictions[runner.name]
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            marker=runner.marker,
            linestyle=runner.linestyle,
            linewidth=2,
            markersize=4,
            color=runner.color,
            label=f"{runner.label} generated",
        )
        ax.scatter(pred[-1, 0], pred[-1, 1], c=runner.color, s=45, label=f"{runner.label} end")

    ax.set_title(title)
    ax.set_xlabel("x / forward")
    ax.set_ylabel("y / lateral")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def selected_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return ["flow", "regression", "diffusion"]
    return [model_arg]


def checkpoint_for_model(args: argparse.Namespace, model_name: str) -> Path:
    if args.checkpoint is not None and args.model != "all":
        return Path(args.checkpoint)
    explicit_paths = {
        "flow": args.flow_checkpoint,
        "regression": args.regression_checkpoint,
        "diffusion": args.diffusion_checkpoint,
    }
    return Path(explicit_paths[model_name])


def command_ids_from_args(command_id: int | None, vocab: list[str]) -> list[int]:
    if command_id is None:
        return list(range(len(vocab)))
    if command_id < 0 or command_id >= len(vocab):
        raise ValueError(f"command id 超出范围：{command_id}")
    return [command_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy_flow.yaml")
    parser.add_argument("--model", type=str, default="flow", choices=MODEL_CHOICES)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--flow-checkpoint", type=str, default=str(DEFAULT_CHECKPOINTS["flow"]))
    parser.add_argument("--regression-checkpoint", type=str, default=str(DEFAULT_CHECKPOINTS["regression"]))
    parser.add_argument("--diffusion-checkpoint", type=str, default=str(DEFAULT_CHECKPOINTS["diffusion"]))
    parser.add_argument("--output-dir", type=str, default="outputs/toy_flow/figures")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--command-id", type=int, default=None)
    parser.add_argument("--progress-weight", type=float, default=0.05)
    parser.add_argument("--lane-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.5)
    args = parser.parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples 必须大于等于 1")

    base_cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    seed = int(base_cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = device_from_cfg(base_cfg["training"].get("device", "auto"))

    runners = []
    for model_name in selected_models(args.model):
        checkpoint_path = checkpoint_for_model(args, model_name)
        try:
            runners.append(build_runner(model_name, checkpoint_path, base_cfg, device))
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            if args.model == "all":
                print(f"跳过 {model_name}：{exc}")
                continue
            raise
    if not runners:
        raise RuntimeError("没有可用的轨迹生成模型，请检查 checkpoint 路径或模型实现")

    ds = build_dataset(base_cfg, seed=seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = base_cfg["data"]["language_vocab"]
    command_ids = command_ids_from_args(args.command_id, vocab)
    reward_weights = RewardWeights(
        progress=args.progress_weight,
        lane=args.lane_weight,
        smoothness=args.smoothness_weight,
    )
    suffix = args.model if args.model != "all" else "all"
    if args.num_samples > 1 and any(runner.name == "flow" for runner in runners):
        suffix = f"{suffix}_k{args.num_samples}_ranked"

    for cmd_id in command_ids:
        command = vocab[cmd_id]
        sample = find_sample_for_command(ds, cmd_id)
        gt = sample["trajectory"].cpu().numpy()
        predictions = {}
        flow_candidates = None
        best_flow_idx = None
        for runner in runners:
            state = sample["state"].unsqueeze(0).to(device)
            cmd = sample["cmd_id"].unsqueeze(0).to(device)
            if runner.name == "flow" and args.num_samples > 1:
                candidates = runner.predict_candidates(state, cmd, args.num_samples)
                lane_center = lane_center_for_command(cmd_id)
                best_idx, scores, components = rank_trajectories(candidates, lane_center=lane_center, weights=reward_weights)
                flow_candidates = candidates.cpu().numpy()
                best_flow_idx = best_idx
                predictions[runner.name] = flow_candidates[best_idx]
                print({
                    "command_id": cmd_id,
                    "model": runner.name,
                    "best_idx": best_idx,
                    "best_reward": round(float(scores[best_idx].cpu()), 4),
                    "progress": round(float(components["progress"][best_idx].cpu()), 4),
                    "lane_deviation": round(float(components["lane_deviation"][best_idx].cpu()), 4),
                    "smoothness": round(float(components["smoothness"][best_idx].cpu()), 4),
                })
            else:
                pred = runner.predict(state, cmd)
                predictions[runner.name] = pred.squeeze(0).cpu().numpy()

        out_path = out_dir / f"command_{cmd_id}_{suffix}.png"
        plot_trajectories(
            gt,
            predictions,
            runners,
            f"Command {cmd_id}: {command}",
            out_path,
            flow_candidates=flow_candidates,
            best_flow_idx=best_flow_idx,
        )
        print(f"已保存轨迹图：{out_path}")


if __name__ == "__main__":
    main()
