# Imports and obligatory syntax
from flask import Flask, request, render_template, jsonify, Response
import threading, json
from queue import Queue, Empty
import static.scripts.forecast_sc as forecast_sc
app = Flask(__name__)

# Script routes
@app.route("/run-forecast", methods=["POST"])
# Begins the training process when the user enters a ticker and presses go
def run_script():
    # Retrieve value for ticker variable
    payload = request.get_json(silent=True) or {}
    ticker = payload.get("ticker", "AAPL") #TODO check legitimacy of APPL

    # Call forecast_sc.py to run off main thread
    thread = threading.Thread(
        target=forecast_sc.run,
        args=(ticker,),
        daemon=True  # So as to not block shutdown
    )
    thread.start()
    return jsonify(status="started", ticker=ticker), 202

@app.route("/check-forecast")
# Used to monitor if forecast_sc.is_complete, and returns its output if so
def check_forecast():
    return jsonify(
        ready=bool(forecast_sc.is_complete),
        output=str(forecast_sc.output or ""),
        pstring=str(forecast_sc.pstring)
    )

@app.route('/display-forecast-plot')
# When forecast_sc.is_complete, this will be used to plot something or not
def show_plot():
    # TODO reconsider first if
    if (not forecast_sc.is_done()):
        return "Not ready", 202
    elif forecast_sc.y_preds_train is None or forecast_sc.y_preds_test is None: # TODO
        # TODO display whoops sorry.img
        return "Forecast data incomplete, cannot plot.", 500
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
