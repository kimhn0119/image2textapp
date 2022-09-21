import os
import requests
import json
from io import BytesIO

from flask import Flask, jsonify, render_template, request, send_file

from modules.inference import infer_t5
from modules.dataset import query_emotion

# https://huggingface.co/settings/tokens
# https://huggingface.co/spaces/{username}/{space}/settings
#API_TOKEN = os.getenv("BIG_GAN_TOKEN")

API_TOKEN = os.getenv('github')

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/infer_biggan")
def biggan():
    input = request.args.get("input")

    output = requests.request(
        "POST",
        "https://api-inference.huggingface.co/models/osanseviero/BigGAN-deep-128",
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        data=json.dumps(input),
    )

    return send_file(BytesIO(output.content), mimetype="image/png")


@app.route("/infer_t5")
def t5():
    input = request.args.get("input")

    output = infer_t5(input)

    return jsonify({"output": output})


@app.route("/query_emotion")
def emotion():
    start = request.args.get("start")
    end = request.args.get("end")

    print(start)
    print(end)

    output = query_emotion(int(start), int(end))

    return jsonify({"output": output})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)