// This script is solely used to provide helper functions used to mediate behavior between forecast_sc.py and index.html
// All actual functionaliy of the forecasting widgit can be found in forecast_sc.py
function runForecast() {
    // Update appearance with loading assets
    const img = document.getElementById('forecast-plot')
    img.src = 'static/img/spinner.gif'
    // Display promise message and pass to check result
    document.getElementById('lstm-target').textContent = "Script started, please wait..."
    fetch('/run-forecast').then(() => setTimeout(checkResult, 1000))
}

function checkResult() {
  fetch('/display-forecast')
    .then(r => r.json())
    .then(data => {
      if (data.ready) {
        // TODO update this element contiuously (place in else chunk)
        document.getElementById('lstm-target').textContent = data.output
        // Uptdate the html img element with the plot
        const img = document.getElementById('forecast-plot')
        img.src = `/display-forecast-plot?ts=${Date.now()}` // Cache-bust
      } else {
        setTimeout(checkResult, 3000)
      }
    })
    .catch(err => {
      document.getElementById('lstm-target').textContent = "Error fetching output."
      console.error(err)
    })
}