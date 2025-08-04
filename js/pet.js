// asdgkhakdghghljak TODO annotate this page
let seg_session
const HW = 256

function randomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// On page load, display a random pet image-segmentation pair
const pet = document.getElementById('pet-img')
const map = document.getElementById('pet-map')

let rand = randomInt(1, 8);
let s = `meta/pets/RGB_${rand}.png`
pet.src = s
s = `meta/pets/heatmap_${rand}.png`
map.src = s

// Handle upload button
document.getElementById('pet-upload').addEventListener('change', function(event) {
    // Change pet image to user image, return early if not possible
    const file = event.target.files[0]
    if (!file) return 
    const imgElement = document.getElementById('pet-img');
    imgElement.src = URL.createObjectURL(file);

    // Process user image into input tensor format which is expected by model
    imgElement.onload = async () => {
        // Early stopping condition in case model failed to load
        const segElement = document.getElementById('pet-map')
        if (!seg_session) {
            // sorry bud but as an image
            console.error('ONNX seg_session not initialized yet.')
        } else {
            URL.revokeObjectURL(imgElement.src) // free memory
            // Draw image to undisplayed canvas of appropriate dimensions
            const canvas = document.getElementById('pet-canvas')
            let ctx = canvas.getContext('2d')
            ctx.drawImage(imgElement, 0, 0, HW, HW)
            let imageData = ctx.getImageData(0, 0, HW, HW).data;

            // Convert img data to float32 array (CHW)
            const data = new Float32Array(3 * HW * HW);
            for (let i = 0; i < HW * HW; i++) {
                const r = imageData[i * 4] / 255;
                const g = imageData[i * 4 + 1] / 255;
                const b = imageData[i * 4 + 2] / 255;

                data[i] = r;
                data[i + HW * HW] = g;
                data[i + 2 * HW * HW] = b;
            }
            // Convert to tensor expected by onnx model    
            const inputT = new ort.Tensor('float32', data, [1, 3, HW, HW]);

            // Perform inference on the input tensor
            const feeds = { [seg_session.inputNames[0]]: inputT }
            const results = await seg_session.run(feeds)
            const output = results[seg_session.outputNames[0]]
            const outputData = output.data; // convert to Float32Array as expected by canvas

            // Update the seg image with the model's inference
            imageData = ctx.createImageData(HW, HW);
            // Colormap for 3 classes
            const colors = [
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
                    const val = outputData[c * HW * HW + i]; // channel-major offset
                    if (val > maxVal) {
                        maxVal = val;
                        maxClass = c;
                    }
                }
                const [r, g, b] = colors[maxClass];
                imageData.data[i * 4 + 0] = r;
                imageData.data[i * 4 + 1] = g;
                imageData.data[i * 4 + 2] = b;
                imageData.data[i * 4 + 3] = 255;
            }
            // Convert to <img> format by resuing canvas, ctx, and imageData elements
            ctx.putImageData(imageData, 0, 0);
            const imgURL = canvas.toDataURL();
            segElement.src = imgURL; // Finally display image on site
        }
    }
})


// ONNX functions
async function initPetModel() {
    // load the model, called once only
    try {
        seg_session = await ort.InferenceSession.create('meta/models/seg_model.onnx');
        console.log('Seg model loaded successfully.')
    } catch (error) {
        //TODO document.getElementById('seg_model_err').textContent()
        console.error('Failed to load ONNX model:', error)
    }
}


window.addEventListener('load', async () => {
    await initPetModel()
})