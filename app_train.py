from flask import Flask, render_template, request, jsonify
import subprocess, os, sys

app_train = Flask(__name__)

# Default and editable args
DEFAULT_ARGS = {
    "dataset_repo_id": "koch_test",
    "policy": "act_koch_real",
    "env": "koch_real",
    "hydra.run.dir": "outputs/train/act_koch_test",
    "hydra.job.name": "act_koch_test",
    "device": "cuda",
    "wandb.enable": "false"
}

EDITABLE_KEYS = ["dataset_repo_id", "device"]

@app_train.route('/')
def home():
    return render_template('train.html')

@app_train.route('/get_train_args', methods=['POST'])
def get_train_args():
    editable_args = {k: DEFAULT_ARGS[k] for k in EDITABLE_KEYS}
    return jsonify(editable_args)

@app_train.route('/run_train', methods=['POST'])
def run_train():
    user_args = request.json.get("args", {})
    full_args = DEFAULT_ARGS.copy()
    full_args.update(user_args)

    script_path = os.path.abspath("lerobot/scripts/train.py")
    cmd = [sys.executable, script_path]
    for key, value in full_args.items():
        cmd.append(f"{key}={value}")

    print("Running command:", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".") + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(" ".join(cmd), capture_output=True, text=True, shell=True, env=env)
        return jsonify({"output": result.stdout, "error": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app_train.run(debug=True, port=5001)
