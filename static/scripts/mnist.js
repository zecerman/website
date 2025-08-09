// TODO annotate this page
// Global parameters
const SIZE = 28
let data   = new Float32Array(1 * 1 * 28 * 28)
let mnist_session 

// Board functions
function initBoard() {
    // create the grid and set its dimensions
    let board = document.querySelector('#board')
    board.style.gridTemplateRows = `repeat(${SIZE}, 8px)`
    board.style.gridTemplateColumns = `repeat(${SIZE}, 8px)`
    // populate the grid
    for (let i = 0; i < SIZE**2; i++) {
        let square = document.createElement('div')
        // color squares white, and listening to be moused over or clicked
        square.style.backgroundColor = 'white'
        square.addEventListener('pointerover', colorSquare)
        square.addEventListener('pointerdown', colorSquare)
        // insert the square into the grid
        board.appendChild(square)
    }
        // print to log if working
        console.log("Board initialized with", board.children.length, "pixels")
}

function initGuessBars() {
  const chart = document.getElementById('guess-chart')
  for (let i = 0; i < 10; i++) {
    const barContainer = document.createElement('div')
    barContainer.style.display = 'flex'
    barContainer.style.flexDirection = 'column'
    barContainer.style.alignItems = 'center'
    barContainer.style.width = '24px'
    barContainer.style.height = '100%' 

    const bar = document.createElement('div')
    bar.className = 'guess-bar'
    bar.style.width = '100%'
    bar.style.height = '0%' // Will be updated dynamically
    bar.style.backgroundColor = '#ffc107'
    bar.style.transition = 'height 0.3s'

    const label = document.createElement('div')
    label.className = 'guess-label'

    label.innerText = i

    barContainer.appendChild(bar)
    barContainer.appendChild(label)
    chart.appendChild(barContainer)
  }
}

function colorSquare(e) {
    // if m1 is down, then color black
    if (e.buttons === 1 || e.type === 'pointerdown') {
        this.style.backgroundColor = 'black'
    }
    checkBoard()
}

function resetBoard() {
    document.querySelectorAll('#board div')       // all squares in the grid
        .forEach(e => e.style.backgroundColor = 'white')
    checkBoard()
}

// ONNX functions
async function initMNISTModel() {
  // load the model, called once only
  try {
    mnist_session = await ort.InferenceSession.create(MNIST_PATH)
    console.log('Mnist model loaded successfully.')
  } catch (error) {
    console.error('Failed to load ONNX model:', error)
  }
}

async function checkBoard() {
    const allDivs = document.getElementById('board').querySelectorAll('div')
    // update data array to reflect board state
    allDivs.forEach((div, idx) => {
        const color = window.getComputedStyle(div).backgroundColor
        data[idx] = (color === 'rgb(0, 0, 0)') ? 255 : 0
    })
    // update onnx tensor to match data array
    const inputT = new ort.Tensor('float32', data, [1, 1, 28, 28])

    if (mnist_session) {
        // feed input tensor, perform inference, save to output
        const feeds = { [mnist_session.inputNames[0]]: inputT }
        const results = await mnist_session.run(feeds)
        const output = results[mnist_session.outputNames[0]]
        // update guess w/ new correct probs
        const probs = softmax(output.data)
        updateGuess(probs)
    } else {
        console.error('ONNX mnist_session failed to initialize.')
    }
    
  }

// Helpers
function softmax(arr) {
    const max = Math.max(...arr)
    const exps = arr.map(x => Math.exp(x - max))
    const sum = exps.reduce((a, b) => a + b, 0)
    return exps.map(x => x / sum)
}

function updateGuess(probs) {
  const bars = document.querySelectorAll('.guess-bar')

  probs.forEach((prob, i) => {
    const percent = Math.round(prob * 100)
    bars[i].style.height = percent + '%'
  })
}

window.addEventListener('load', async () => {
  initBoard()
  initGuessBars()
  await initMNISTModel()

  // so that mouseover is not necessary to display predictions
  setInterval(checkBoard, 500)
})