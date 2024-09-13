import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "App is working!"

if __name__ == "__main__":
    # Ensure Flask binds to the correct port
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
