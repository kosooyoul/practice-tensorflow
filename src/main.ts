import * as tf from '@tensorflow/tfjs';

const SEQ_LENGTH = 5; // 입력 시퀀스 길이

async function testModel(model: tf.LayersModel, testText: string) {
  const testSequence = testText.split('').map(c => c.charCodeAt(0) / 255);
  const xTest = tf.tensor2d([testSequence], [1, SEQ_LENGTH]);

  const prediction = model.predict(xTest) as tf.Tensor;
  const predictedValue = (await prediction.data())[0] * 255;
  const predictedChar = String.fromCharCode(predictedValue);

  return predictedChar;
}

async function main() {
  // 딥러닝 모형 정의
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: 20000, outputDim: 300, inputLength: SEQ_LENGTH }));
  model.add(tf.layers.lstm({ units: 200, inputShape: [SEQ_LENGTH, 1], returnSequences: false }));
  model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
  model.compile({ optimizer: 'adam', loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
  model.summary();

  console.log(model.summary());

  // 딥러닝 학습
  const sequences = [
    { input: 'chips', target: '0' },
    { input: 'candy', target: '0' },
    { input: 'berry', target: '1' },
    { input: 'apple', target: '0' },
    { input: 'caida', target: '0' },
    { input: 'milks', target: '0' },
    { input: 'rices', target: '1' },
    { input: 'ramen', target: '1' },
  ];

  const x = tf.tensor2d(
    sequences.map(s => s.input.split('').map(c => c.charCodeAt(0) / 255)),
    [sequences.length, SEQ_LENGTH],
  );
  const y = tf.tensor1d(sequences.map(s => s.target.charCodeAt(0) / 255));
  const history = await model.fit(x, y, { epochs: 10, batchSize: 128 });

  console.log(history);

  console.log('start test');
  const result = await testModel(model, 'apple');

  console.log('test result is ' + result);
}

main();
