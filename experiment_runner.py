"""Воспроизводимые эксперименты Практики 2 для GPU Google Colab.

Запускать после установки зависимостей из practice2.ipynb.
Источником обучения служит train.ipynb учебного репозитория.
Существующие завершённые эксперименты повторно не запускаются.
"""

from __future__ import annotations

import csv
import gc
import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import urllib.request

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SOURCE_URL = "https://raw.githubusercontent.com/Yandex-Practicum/ap-physicalai-1/main/train.ipynb"
EXPERIMENTS = [
    ("baseline", 1001, None, None),
    ("max_iters_500", 500, None, None),
    ("box_target_strong", 1001, "box_target", 40.0),
    ("qpos_strong", 1001, "robot_target_qpos", 30.0),
    ("no_floor_collision_x10", 1001, "no_floor_collision", 2.5),
]
HYPOTHESIS = {
    "parameter": "no_floor_collision",
    "baseline_value": 0.25,
    "experiment_value": 2.5,
    "expected_mean_reward": "increase",
    "expected_SR": "decrease",
    "reason": (
        "Больший вес вспомогательного поощрения за отсутствие контакта с полом "
        "может увеличить итоговую награду, но стимулировать избегание движений "
        "к кубу и снизить долю успешных доставок."
    ),
}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def flush_writer(context: dict, close: bool = False) -> None:
    runner = context.get("runner")
    for owner in (getattr(runner, "logger", None), runner):
        writer = getattr(owner, "writer", None)
        if writer is not None:
            writer.flush()
            if close:
                writer.close()


def evaluate(context: dict, record_video: bool = True) -> dict:
    """10 независимых reset, seed 42–51, как в inference.py курса."""
    runner = context["runner"]
    env = context["raw_env"]
    config = context["env_cfg"]
    wrapper_torch = context["wrapper_torch"]
    log_dir = Path(context["log_dir"])
    policy = runner.get_inference_policy(device=context["device"])
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    episodes = []
    video_path = log_dir / "evaluation.mp4"
    video_error = None

    for episode in range(10):
        state = reset(jax.random.PRNGKey(42 + episode))
        trajectory = [state] if episode == 0 and record_video else None
        total_reward = 0.0
        max_cube_z = float(state.data.qpos[11])
        for index in range(config.episode_length):
            observation = state.obs["state"] if isinstance(state.obs, dict) else state.obs
            with torch.no_grad():
                action = policy({"state": wrapper_torch._jax_to_torch(observation)})
                action = torch.clip(action, -1.0, 1.0)
            state = step(state, wrapper_torch._torch_to_jax(action.flatten()))
            total_reward += float(state.reward)
            max_cube_z = max(max_cube_z, float(state.data.qpos[11]))
            if trajectory is not None and index % 4 == 0:
                trajectory.append(state)
            if bool(state.done):
                break
        distance = float(np.linalg.norm(
            np.asarray(state.info["target_pos"]).flatten()
            - np.asarray(state.data.qpos[9:12])
        ))
        record = {
            "episode": episode + 1,
            "seed": 42 + episode,
            "reward": total_reward,
            "max_cube_z": max_cube_z,
            "final_distance": distance,
            "success": distance < 0.05,
            "grasped": max_cube_z > 0.2,
        }
        episodes.append(record)
        print("EVALUATION", record, flush=True)
        if trajectory is not None:
            try:
                import imageio.v2 as imageio
                frames = env.render(trajectory, height=360, width=480)
                imageio.mimsave(video_path, frames, fps=max(1, round(1 / env.dt / 4)))
            except Exception as error:
                # Ошибка видео не должна превращаться в выдуманный результат SR.
                video_error = repr(error)
                print("VIDEO_ERROR", video_error, flush=True)

    flush_writer(context)
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    events.Reload()
    tags = events.Tags()["scalars"]
    candidates = [tag for tag in tags if tag == "Train/mean_reward"]
    if not candidates:
        candidates = [tag for tag in tags if "mean_reward" in tag.lower() and "time" not in tag.lower()]
    curve = [(point.step, point.value) for point in events.Scalars(candidates[0])] if candidates else []
    checkpoints = list(log_dir.glob("model_*.pt"))
    checkpoint = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1])) if checkpoints else None
    result = {
        "experiment": context["EXP_NAME"],
        "num_envs": context["NUM_ENVS"],
        "max_iters": context["MAX_ITERS"],
        "training_seconds": context["elapsed"],
        "mean_reward": curve[-1][1] if curve else None,
        "reward_tag": candidates[0] if candidates else None,
        "success_count": sum(record["success"] for record in episodes),
        "episode_count": 10,
        "episodes": episodes,
        "curve": curve,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "video": str(video_path) if video_path.exists() else None,
        "video_error": video_error,
        "reward_scales": config.reward_config.scales.to_dict(),
    }
    write_json(log_dir / "evaluation.json", result)
    print("EXPERIMENT_RESULT", json.dumps({key: value for key, value in result.items() if key not in ("episodes", "curve")}, ensure_ascii=False), flush=True)
    return result


def backup(result: dict, output_dir: Path, download: bool) -> None:
    name = result["experiment"]
    write_json(output_dir / f"{name}.json", result)
    archive = shutil.make_archive(f"{name}_results", "zip", root_dir=".", base_dir=f"logs/{name}")
    print("BACKUP_ARCHIVE", archive, flush=True)
    if download:
        from google.colab import files
        files.download(archive)


def run_all(download: bool = True) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("Для полного запуска выберите T4 GPU в настройках Colab.")
    output_dir = Path("practice2_results")
    write_json(output_dir / "hypothesis.json", HYPOTHESIS)
    versions = {}
    for package in ("torch", "playground", "jax", "jaxlib", "mujoco", "mujoco-mjx", "rsl-rl-lib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(output_dir / "package_versions.json", versions)
    with urllib.request.urlopen(SOURCE_URL) as response:
        notebook = json.load(response)
    source = "".join(notebook["cells"][8]["source"])
    (output_dir / "source_training.py").write_text(source, encoding="utf-8")
    write_json(output_dir / "source.json", {"url": SOURCE_URL, "sha256": hashlib.sha256(source.encode()).hexdigest()})
    context = {"__name__": "__practice2_training__"}
    results = []

    for name, iterations, reward_key, reward_value in EXPERIMENTS:
        result_file = Path("logs") / name / "evaluation.json"
        if result_file.exists():
            result = json.loads(result_file.read_text(encoding="utf-8"))
            print("USING_COMPLETED_EXPERIMENT", name, flush=True)
        else:
            if glob.glob(f"logs/{name}/model_*.pt"):
                raise RuntimeError(f"Есть незавершённый запуск logs/{name}. Сначала оцените или продолжите его, чтобы не перезаписать чекпоинты.")
            flush_writer(context, close=True)
            context.clear()
            context["__name__"] = "__practice2_training__"
            gc.collect()
            torch.cuda.empty_cache()
            jax.clear_caches()
            code = source.replace("EXP_NAME = None", f"EXP_NAME = {name!r}")
            code = code.replace("MAX_ITERS = 1001", f"MAX_ITERS = {iterations}")
            if reward_key is not None:
                code = code.replace(
                    "env_cfg = registry.get_default_config(ENV_NAME)",
                    "env_cfg = registry.get_default_config(ENV_NAME)\n"
                    f"env_cfg.reward_config.scales.{reward_key} = {reward_value!r}",
                )
            print("STARTING_EXPERIMENT", name, flush=True)
            exec(compile(code, f"<practice2_{name}>", "exec"), context)
            result = evaluate(context)
        results.append(result)
        backup(result, output_dir, download)
        write_json(output_dir / "all_results.json", results)

    with (output_dir / "observations.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "num_envs", "max_iters", "mean_reward", "SR", "training_seconds", "grasped_episodes", "checkpoint"])
        for result in results:
            writer.writerow([result["experiment"], result["num_envs"], result["max_iters"], result["mean_reward"], f'{result["success_count"]}/10', result["training_seconds"], sum(episode["grasped"] for episode in result["episodes"]), result["checkpoint"]])
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 5))
    for result in results:
        if result["curve"]:
            x, y = zip(*result["curve"])
            axis.plot(x, y, label=result["experiment"], alpha=0.8)
    axis.set(xlabel="Iteration", ylabel="Training mean reward", title="PandaPickCube learning curves")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(figure)
    archive = shutil.make_archive("practice2_summary", "zip", output_dir)
    if download:
        from google.colab import files
        files.download(archive)
    print("PRACTICE2_ALL_EXPERIMENTS_COMPLETE", flush=True)
    return results


if __name__ == "__main__":
    run_all()
