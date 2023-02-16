import * as tf from '@tensorflow/tfjs';

const SEQ_LENGTH = 5; // 입력 시퀀스 길이

async function main() {
  // 딥러닝 모형 정의
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: 20000, outputDim: 300, inputLength: 5 }));
  model.add(tf.layers.lstm({ units: 200, inputShape: [SEQ_LENGTH, 1], returnSequences: false }));
  model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
  model.compile({ optimizer: 'adam', loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
  model.summary();

  console.log(model.summary());

  // 딥러닝 학습
  const sequences = [
    { input: 'chips', target: 'snack' },
    { input: 'candy', target: 'snack' },
    { input: 'berry', target: 'fruit' },
    { input: 'apple', target: 'fruit' },
    { input: 'caida', target: 'drink' },
    { input: 'milks', target: 'drink' },
    { input: 'rices', target: 'foods' },
    { input: 'ramen', target: 'foods' },
  ];

  const x = tf.tensor2d(
    sequences.map(s => s.input.split('').map(c => c.charCodeAt(0) / 255)),
    [sequences.length, SEQ_LENGTH],
  );
  const y = tf.tensor1d(sequences.map(s => s.target.charCodeAt(0) / 255));
  const history = await model.fit(x, y, { epochs: 10, batchSize: 128 });

  console.log(history);
}

main();
