from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello! The server is working!"

if __name__ == '__main__':
    print("\nServer is running! Access it at: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, port=5000)
