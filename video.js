// Get the video and canvas elements
const video = document.getElementById('videoInput');
const canvasOutput = document.getElementById('canvasOutput');
const ctx = canvasOutput.getContext('2d');

// Create OpenCV mats for processing
const src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
const dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);

// Create video capture object
const cap = new cv.VideoCapture(video);

// Boolean flag to control video streaming
let streaming = true;

// Function to process the video frames
function processVideo() {
  try {
    if (!streaming) {
      // Clean up and stop if not streaming
      src.delete();
      dst.delete();
      return;
    }

    // Start processing
    cap.read(src);
    cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);

    // Render the processed frame on the canvas
    ctx.drawImage(dst, 0, 0, canvasOutput.width, canvasOutput.height);

    // Schedule the next frame processing
    requestAnimationFrame(processVideo);
  } catch (err) {
    console.error(err);
  }
}

// Start the video processing
processVideo();
