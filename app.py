from flask import Flask, render_template, request, jsonify
import subprocess, os, sys, signal
from pynput.keyboard import Controller, Key
from datetime import datetime

app = Flask(__name__)
process = None  # Store the current subprocess globally

# ---------- Control Defaults ----------
DEFAULT_ARGS = {
    "record": {
        "robot-path": "lerobot/configs/robot/koch.yaml",
        "fps": 30,
        "root": "demo_data",
        "repo-id": "koch_test",
        "tags": "tutorial",
        "warmup-time-s": 5,
        "episode-time-s": 30,
        "reset-time-s": 30,
        "num-episodes": 2,
        "push-to-hub": 0
    },
    "record-with-marker": {
        "robot-path": "lerobot/configs/robot/koch.yaml",
        "fps": 30,
        "root": "demo_data",
        "repo-id": "koch_test",
        "tags": "marker",
        "warmup-time-s": 5,
        "episode-time-s": 30,
        "reset-time-s": 30,
        "num-episodes": 2,
        "push-to-hub": 0
    },
    "replay": {
        "robot-path": "lerobot/configs/robot/koch.yaml",
        "fps": 30,
        "root": "demo_data",
        "repo-id": "koch_test",
        "episode": 0
    },
    "teleoperate": {
        "robot-path": "lerobot/configs/robot/koch.yaml"
    }
}

EDITABLE_KEYS = {
    "record": ["repo-id", "num-episodes"],
    "record-with-marker": ["repo-id", "num-episodes"],
    "replay": ["repo-id", "episode"],
    "teleoperate": []
}

# ---------- Training Defaults ----------
TRAIN_DEFAULT_ARGS = {
    "dataset_repo_id": "koch_test",
    "policy": "act_koch_real",
    "env": "koch_real",
    "hydra.run.dir": "demo_outputs/train/act_koch_test",
    "hydra.job.name": "act_koch_test",
    "device": "cuda",
    "wandb.enable": "false"
}

TRAIN_EDITABLE_KEYS = ["dataset_repo_id", "device"]

# ---------- Routes ----------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/get_args', methods=['POST'])
def get_args():
    mode = request.json.get("mode")
    args = DEFAULT_ARGS.get(mode, {})
    editable_keys = EDITABLE_KEYS.get(mode, [])
    editable_args = {k: args[k] for k in editable_keys if k in args}
    return jsonify(editable_args)

@app.route('/run_script', methods=['POST'])
def run_script():
    global process
    data = request.json
    mode = data.get("mode")
    user_args = data.get("args", {})

    full_args = DEFAULT_ARGS.get(mode, {}).copy()

    # Modify repo-id to include datetime if present and editable
    if "repo-id" in EDITABLE_KEYS.get(mode, []) and "repo-id" in user_args:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        user_args["repo-id"] = f"{user_args['repo-id']}_{timestamp}"

    full_args.update(user_args)

    script_path = os.path.abspath("lerobot/scripts/control_robot.py")
    cmd = [sys.executable, script_path, mode.replace("-", "_")]
    for key, value in full_args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    print("Running command:", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".") + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        return jsonify({"output": result.stdout, "error": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/send_key', methods=['POST'])
def send_key():
    keyboard = Controller()
    key = request.json.get("key")
    key_map = {
        "ArrowLeft": Key.left,
        "ArrowRight": Key.right,
        "Escape": Key.esc
    }
    if key in key_map:
        keyboard.press(key_map[key])
        keyboard.release(key_map[key])
        return jsonify({"status": "success", "key": key})
    else:
        return jsonify({"status": "error", "message": "Invalid key"}), 400

@app.route('/stop_script', methods=['POST'])
def stop_script():
    global process
    if process and process.poll() is None:
        os.kill(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        return jsonify({"status": "terminated"})
    return jsonify({"status": "no_process"})

@app.route('/get_train_args', methods=['POST'])
def get_train_args():
    editable_args = {k: TRAIN_DEFAULT_ARGS[k] for k in TRAIN_EDITABLE_KEYS}
    return jsonify(editable_args)

@app.route('/run_train', methods=['POST'])
def run_train():
    user_args = request.json.get("args", {})
    train_type = request.json.get("train_type", "train")  # Default fallback
    full_args = TRAIN_DEFAULT_ARGS.copy()
    full_args.update(user_args)

    # Dynamically update dependent keys
    if "dataset_repo_id" in user_args:
        dataset_name = user_args["dataset_repo_id"]
        # Use selected train_type in the run directory
        full_args["hydra.run.dir"] = f"outputs/{train_type}/act_{dataset_name}"
        full_args["hydra.job.name"] = f"act_{dataset_name}"

    script_path = os.path.abspath("lerobot/scripts/train.py")
    # cmd = [sys.executable, script_path]
    cmd = ["DATA_DIR=data", sys.executable, script_path]
    for key, value in full_args.items():
        cmd.append(f"{key}={value}")

    print("Running command:", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".") + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(" ".join(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, env=env)
        return jsonify({"output": result.stdout, "error": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)})

# ---------- Main ----------
if __name__ == '__main__':
    app.run(debug=True)
