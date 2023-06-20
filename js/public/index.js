const problemInput = document.getElementById("problem")
const canvas = document.getElementById("solution")
const checkAnswerButton = document.getElementById("buttonCheck")
const statusText = document.getElementById("statusText")
const ctx = canvas.getContext("2d")
const STATUS_CORRECT = "Correct! ✅"
const STATUS_INCORRECT = "Incorrect! ❌"
let isDrawing = false
let lastX = 0
let lastY = 0
let coord = { x: 0, y: 0 }

document.addEventListener("mousedown", start)
document.addEventListener("mouseup", stop)

function reposition(event) {
    coord.x = event.clientX - canvas.offsetLeft
    coord.y = event.clientY - canvas.offsetTop
}

function start(event) {
    document.addEventListener("mousemove", draw)
    reposition(event)
}

function stop() {
    document.removeEventListener("mousemove", draw)
}

function draw(event) {
    ctx.beginPath()
    ctx.lineWidth = 6
    ctx.lineCap = "round"
    ctx.strokeStyle = "#000000"
    ctx.moveTo(coord.x, coord.y)
    reposition(event)
    ctx.lineTo(coord.x, coord.y)
    ctx.stroke()
}

function checkResult() {
    let image = new Image()
    let problem = problemInput.value
    image.id = "result"
    image.src = canvas.toDataURL()
}
