// This script is solely used to provide helper functions used to mediate behavior between forecast_sc.py and index.html
// All actual functionaliy of the forecasting widgit can be found in forecast_sc.py


async function runForecast(ticker) {
    // Update appearance with loading assets
    const img = document.getElementById('forecast-plot')
    img.src = '/static/img/spinner.gif'
    img.style = 'width:250px;height:250px;'
    // Tell app.py to execute run-forecast
    const res = await fetch('/run-forecast', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ ticker })
    });

    // Display promise message and pass to check result
    document.getElementById('lstm-target').textContent = `Model fitting to ${ticker}'s data, please wait...`
    setTimeout(checkResult(ticker), 10)
}

function checkResult(ticker) {
  bar = document.getElementById('progress-bar')
  bar.style = 'display:flex;justify-content:center;'
  fetch('/check-forecast')
    .then(r => r.json())
    .then(data => {
      if (data.ready) {
        document.getElementById('lstm-target').textContent = data.output
        // Uptdate the html img element with the plot or lack of plot
        const img = document.getElementById('forecast-plot')
        if (data.output == 'No data found, is your ticker mispelled?') {
          img.src = '/static/img/error.png'
          img.style = 'width:600px;height:250px;' // Restore original size
        } else {
          img.src = `/display-forecast-plot?ts=${Date.now()}` // Cache-bust
          img.style = 'width:600px;height:250px;' // Restore original size
        } 
        bar.style = 'display:none'
      } else {
        bar.textContent = data.pstring
        setTimeout(checkResult, 500)
      }
    })
    .catch(err => {
      document.getElementById('lstm-target').textContent = 'Error fetching output.'
      console.error(err)
    })
}