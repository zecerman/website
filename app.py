from flask import Flask, render_template
import yfinance as yf

app = Flask(__name__)
# Necessary syntax ^

# Function behavior
def get_ticker():
    data = yf.download('MSFT', start='2000-01-01', end='2000-02-02')
    return data.values

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cube')
def cube():
    return render_template('ascii_cube.html')

@app.route('/recipes')
def recipes():
    return render_template('recipes.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/test')
def test():
    return render_template('test.html')

# Main
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
