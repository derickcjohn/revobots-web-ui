from flask import Flask, render_template, request, jsonify
import subprocess, signal
import os
import sys

app = Flask(__name__)
process = None  # Store the current subprocess globally

# Define default arguments for each mode
DEFAULT_ARGS = {
    "record": {
        "robot-path": "lerobot/configs/robot/koch.yaml",
        "fps": 30,
        "root": "data",
        "repo-id": "koch_test",
        "tags": "tutorial",
        "warmup-time-s": 5,
        "episode-time-s": 30,
        "reset-time-s": 30,
        "num-episodes": 2
    },
    "record-with-marker": {
        "robot-path": "lerobot/configs/robot/koch.yaml",
        "fps": 30,
        "root": "data",
        "repo-id": "koch_test",
        "tags": "marker",
        "warmup-time-s": 5,
        "episode-time-s": 30,
        "reset-time-s": 30,
        "num-episodes": 2,

    },
    "replay": {
        "robot-path": "lerobot/configs/robot/koch.yaml",
        "fps": 30,
        "root": "data",
        "repo-id": "koch_test",
        "episode": 0
    },
    "teleoperate": {
        "robot-path": "lerobot/configs/robot/koch.yaml"
    }
}

# Define which args should be shown to the user per mode
EDITABLE_KEYS = {
    "record": ["repo-id", "num-episodes"],
    "record-with-marker": ["repo-id", "num-episodes"],
    "replay": ["repo-id", "episode"],
    "teleoperate": []
}

@app.route('/')
def home():
    return render_template('index.html')

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

    # Combine user args with default args
    full_args = DEFAULT_ARGS.get(mode, {}).copy()
    full_args.update(user_args)

    # Convert args to command-line format
    script_path = os.path.abspath("lerobot/scripts/control_robot.py")
    cmd = [sys.executable, script_path, mode.replace("-", "_")]
    for key, value in full_args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    print("Running command:", " ".join(cmd))

    # Set the PYTHONPATH to the project root so lerobot can be imported
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".") + os.pathsep + env.get("PYTHONPATH", "")

    # Run the command and return output
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        return jsonify({"output": result.stdout, "error": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/send_key', methods=['POST'])
def send_key():
    from pynput.keyboard import Controller, Key
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
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        return jsonify({"status": "terminated"})
    return jsonify({"status": "no_process"})


if __name__ == '__main__':
    app.run(debug=True)
