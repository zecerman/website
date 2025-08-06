// TODO annotate this page
// Globals
let seg_session
const HW = 256


// Page load logic, display a random pet image-segmentation pair as default
const pet = document.getElementById('pet-img')
const map = document.getElementById('pet-map')
let rand = randomInt(1, 8);
let s = `static/pets/RGB_${rand}.png`
pet.src = s
s = `static/pets/heatmap_${rand}.png`
map.src = s

// Handle upload button
document.getElementById('pet-upload').addEventListener('change', function(event) {
    // Change pet image to user image, or return early if not possible
    let file = event.target.files[0]
    if (!file) return 
    let imgElement = document.getElementById('pet-img');
    imgElement.src = URL.createObjectURL(file);

    imgElement.onload = async () => {
        // Draw user image and temp loading gif
        let segElement = document.getElementById('pet-map')
        segElement.src = 'static/img/spinner.gif'

        // Allow one frame to elapse so both imgElement and segElement may update
        await yieldToBrowser()
        
        // Perform inference on the user image using the segmentation model
        inferSegMap(imgElement, segElement)
    }
})

// ONNX functions
async function initPetModel() {
    // load the model, called once only
    try {
        seg_session = await ort.InferenceSession.create(SEG_PATH);
        console.log('Seg model loaded successfully.')
    } catch (error) {
        //TODO document.getElementById('seg_model_err').textContent()
        console.error('Failed to load ONNX model:', error)
    }
}

async function inferSegMap(imgElement, segElement) {
    // Begin by checking an early stopping condition in case model failed to load
    if (!seg_session) {
        // TODO print sorry bud but as an image
        console.error('ONNX seg_session failed to initialize.')
    } else {
        URL.revokeObjectURL(imgElement.src) // Free memory

        // Draw image to undisplayed canvas of appropriate dimensions
        const canvas = document.getElementById('pet-canvas')
        let ctx = canvas.getContext('2d')
        ctx.drawImage(imgElement, 0, 0, HW, HW)
        let imageData = ctx.getImageData(0, 0, HW, HW).data;

        // Convert img data to float32 array (CHW)
        let data = new Float32Array(3 * HW * HW);
        for (let i = 0; i < HW * HW; i++) {
            let r = imageData[i * 4] / 255;
            let g = imageData[i * 4 + 1] / 255;
            let b = imageData[i * 4 + 2] / 255;

            data[i] = r;
            data[i + HW * HW] = g;
            data[i + 2 * HW * HW] = b;

            // Model processing is done on main thread, pasue to allow browser animations to continue
            if (i % 2048 === 0) {
                await yieldToBrowser()
            }
        }
        // Convert to tensor expected by onnx model    
        let inputT = new ort.Tensor('float32', data, [1, 3, HW, HW]);
        let feeds = { [seg_session.inputNames[0]]: inputT }
        
        // Perform inference on the input tensor (this call is computationally expensize, bandaid fix is to buffer before and after)
        await yieldToBrowser()
        let results = await seg_session.run(feeds)
        await yieldToBrowser()
        
        // convert to Float32Array as expected by canvas
        let output = results[seg_session.outputNames[0]]
        let outputData = output.data; 

        // Update the seg image with the model's inference
        imageData = ctx.createImageData(HW, HW);
        // Colormap for 3 classes
        let colors = [
            [0, 0, 0],        // unused
            [255, 255, 0],      // class 1: fuzzy/outline
            [0, 0, 0],      // class 2: background
            [256/2, 256/2, 0]       // class 3: body
        ];
        // Colorize by argmax function
        // For each pixel position
        for (let i = 0; i < HW * HW; i++) {
            let maxClass = 0;
            let maxVal = outputData[i]; // channel 0
            for (let c = 1; c < 4; c++) {
                let val = outputData[c * HW * HW + i]; // channel-major offset
                if (val > maxVal) {
                    maxVal = val;
                    maxClass = c;
                }
            }
            let [r, g, b] = colors[maxClass];
            imageData.data[i * 4 + 0] = r;
            imageData.data[i * 4 + 1] = g;
            imageData.data[i * 4 + 2] = b;
            imageData.data[i * 4 + 3] = 255;

            // Model processing is done on main thread, pasue to allow browser animations to continue
            if (i % 2048 === 0) {
                await yieldToBrowser()
            }
        }
        // Convert to <img> format by resuing canvas, ctx, and imageData elements
        ctx.putImageData(imageData, 0, 0);
        let imgURL = canvas.toDataURL();
        segElement.src = imgURL; // Finally display image on site
    }
}

// Helper functions
function randomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function yieldToBrowser() {
    // Prompts the broswer to display/buffer the next animation frame, used to prevent the site from hanging
    return new Promise(resolve => {
        requestAnimationFrame(() => {
            setTimeout(resolve, 0);
        });
    });
}

// MAIN
window.addEventListener('load', async () => {
    await initPetModel()
})