"""Run the whole pipeline: pose (in batches) -> preprocess -> train -> test.

    python run_all.py                    # everything
    python run_all.py --skip_pose        # skeletons already extracted
    python run_all.py --iters 200        # shorter training
"""
import argparse
import subprocess
import sys

import config
import pose_extract


def run(script, *args):
    print(f"\n=== python {script} {' '.join(args)} ===", flush=True)
    subprocess.run([sys.executable, script, *args], check=True, cwd=str(config.CODE_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_pose", action="store_true")
    ap.add_argument("--batch_videos", type=int, default=config.BATCH_VIDEOS)
    ap.add_argument("--iters", type=int, default=config.TOTAL_ITERS)
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    if not a.skip_pose:
        if not pose_extract.list_videos():
            raise SystemExit(f"No videos in {config.DATASET_DIR}/<person_id>/")
        model = pose_extract.load_model()
        while pose_extract.pending_videos():
            left = pose_extract.run(a.batch_videos, model=model)
            print(f"pending videos: {left}")
        run("check_pose.py")
    run("preprocess.py")
    train_args = ["--iters", str(a.iters), "--batch_size", str(a.batch_size)]
    run("train.py", *train_args, *(["--resume"] if a.resume else []))
    run("test.py")
    print(f"\nAll done. Results are in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()