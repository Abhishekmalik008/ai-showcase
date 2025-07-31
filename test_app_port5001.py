from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def hello():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0;
            }
            .container {
                text-align: center;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
            }
            .success {
                color: #27ae60;
                font-weight: bold;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Server is Running!</h1>
            <div class="success">✓ Successfully connected to the server on port 5001</div>
            <p>This confirms that your Flask server is working correctly.</p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("\nTest server is running on port 5001!")
    print("Access it at: http://localhost:5001")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, port=5001)
