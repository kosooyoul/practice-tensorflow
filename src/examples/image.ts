import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as jpeg from 'jpeg-js';

async function main() {
  const modelPath = './model.json';
  const model = await tf.loadLayersModel(modelPath);

  const image = fs.readFileSync('./image.jpg');
  const pixels = jpeg.decode(image);
  const tensor = tf.browser.fromPixels(pixels).toFloat().expandDims();

  const prediction = model.predict(tensor);

  console.log(prediction, prediction);
  const predictedClass = prediction[0].argMax(1).dataSync()[0];

  console.log('predictedClass', predictedClass);
}

main();
