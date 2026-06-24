from flask import Flask, request

app = Flask(__name__)

@app.route("/smoke-event", methods=["POST"])
def smoke_event():

    data = request.json

    print()
    print("EVENT RECEIVED")
    print(data)

    return {"status": "success"}, 200


app.run(
    host="0.0.0.0",
    port=5000
)