import * as tf from '@tensorflow/tfjs';

// Tensor
// - 벡터와 행렬을 더 높은 차원으로 일반화한 것, 다차원 배열과 유사
// - tf.Tensor { rank: 텐서에 포함된 차원 수, shape: 각 차원의 크기, dtype: 데이터 유형 }
// ref; https://www.tensorflow.org/js/guide/tensors_operations?hl=ko

function summaryTensor(tensor: any): Pick<any, 'rank' | 'shape' | 'dtype'> {
  return {
    rank: tensor.rank,
    shape: tensor.shape,
    dtype: tensor.dtype,
  };
}

const t1 = tf.tensor([
  [1, 2],
  [3, 4],
]);
console.log('- Tensor_1', summaryTensor(t1));
t1.print();

const t2 = tf.tensor([
  [1, 2],
  [3, 4],
  [3, 4],
]);
console.log('- Tensor_2', summaryTensor(t2));
t2.print();

const t3 = tf.tensor([
  [
    [1, 2, 6],
    [2, 4, 7],
  ],
]);
console.log('- Tensor_3', summaryTensor(t3));
t3.print();

const reshaped_t1 = t1.reshape([1, 4]);
console.log('- Reshaped Tensor_1 Array', t1.arraySync(), '->', reshaped_t1.arraySync());
reshaped_t1.print();

const reshaped_t2 = t2.reshape([2, 3]);
console.log('- Reshaped Tensor_2 Data', t2.dataSync(), '->', reshaped_t2.dataSync());
reshaped_t2.print();

const squared_t1 = t1.square();
console.log('- Squared Tensor_1 Array', t1.arraySync(), '->', squared_t1.arraySync());

const squared_t3 = t3.square();
console.log('- Squared Tensor_3 Data', t3.dataSync(), '->', squared_t3.dataSync());

const added_t1 = t1.add(
  tf.tensor([
    [10, 20],
    [30, 40],
  ]),
);
console.log('- Added Tensor_1 Array', added_t1.arraySync());

const t1_max = t1.argMax(0);
console.log('- Tensor_1 Max', t1_max.arraySync());

const t2_max = t2.argMax(0);
console.log('- Tensor_2 Max', t2_max.arraySync());

t1.dispose();
console.log('- Disposed Tensor_1', summaryTensor(t1));
try {
  console.log('- Disposed Tensor_1 Array', t1.arraySync());
} catch (e) {
  console.log('! Error', e.message);
}
try {
  console.log('- Disposed Tensor_1 Data', t1.dataSync());
} catch (e) {
  console.log('! Error', e.message);
}
