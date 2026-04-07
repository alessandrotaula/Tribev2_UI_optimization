from flask import Flask, jsonify, request

app = Flask(__name__)


@app.get("/")
def root():
    return jsonify(
        {
            "name": "Hero Title Brain Analyzer",
            "status": "ok",
            "message": "Vercel deployment is working.",
            "usage": "Use the CLI locally: python main.py --title \"Your hero title\"",
            "query_example": "/health",
        }
    )


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/run")
def run_not_supported():
    title = request.args.get("title")
    return (
        jsonify(
            {
                "ok": False,
                "reason": "This project is optimized for CLI execution, not Vercel serverless runtime.",
                "received_title": title,
                "how_to_run": "python main.py --title \"Your hero title\"",
            }
        ),
        400,
    )
