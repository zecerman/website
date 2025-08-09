# Imports and obligatory syntax
from flask import Flask, render_template, jsonify
import threading
import static.scripts.forecast_sc as forecast_sc
app = Flask(__name__)

# Script routes
@app.route("/run-forecast")
def run_script():
    thread = threading.Thread(target=forecast_sc.run)
    thread.start()
    return jsonify(status="started")


@app.route("/display-forecast")
def display_forecast():
    if forecast_sc.is_done():
        return jsonify(output=forecast_sc.get_output(), ready=True)
    else:
        return jsonify(output="Script is still running...", ready=False)

@app.route('/display-forecast-plot')
def show_plot():
    if not forecast_sc.is_done():
        return "Not ready", 202
    else:
        return forecast_sc.plot_predictions(
            actuals=forecast_sc.y_actuals,
            predictions=forecast_sc.y_projection,
            dates=forecast_sc.extended_dates,
            y_preds_train_len=len(forecast_sc.y_preds_train),
            y_preds_test_len=len(forecast_sc.y_preds_test)
    )



# Hyperlink routes
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
