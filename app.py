from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Welcome to the ML Job Analytics Dashboard!"})

if __name__ == '__main__':
    app.run(debug=True) 