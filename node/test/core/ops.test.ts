import { strict as assert } from 'node:assert';
import mlx, {
  array,
  reshape,
  transpose,
  moveaxis,
  swapaxes,
  add,
  multiply,
  where,
  arange,
  newStream,
  withStream,
  float32,
  float16,
  int32,
  int64,
  uint32,
} from '../../src';

const toArray = (tensor: ReturnType<typeof array>): number[] => tensor.toArray() as number[];

describe('core ops', () => {
  it('reshape matches element order', () => {
    const original = array([1, 2, 3, 4], [2, 2]);
    const reshaped = reshape(original, [4, 1]);
    assert.deepEqual(reshaped.shape, [4, 1]);
    assert.deepEqual(toArray(reshaped), [1, 2, 3, 4]);
  });

  it('transpose without axes reverses dims', () => {
    const original = array([1, 2, 3, 4], [2, 2]);
    const transposed = transpose(original);
    assert.deepEqual(transposed.shape, [2, 2]);
    assert.deepEqual(toArray(transposed), [1, 3, 2, 4]);
  });

  it('transpose with axes reorders dims explicitly', () => {
    const original = array([1, 2, 3, 4, 5, 6], [1, 2, 3]);
    const transposed = transpose(original, [2, 0, 1]);
    assert.deepEqual(transposed.shape, [3, 1, 2]);
  });

  it('moveaxis shifts axes correctly', () => {
    const original = array([1, 2, 3, 4], [2, 2]);
    const moved = moveaxis(original, 0, 1);
    assert.deepEqual(moved.shape, [2, 2]);
    assert.deepEqual(toArray(moved), [1, 3, 2, 4]);
  });

  it('swapaxes exchanges two axes', () => {
    const original = array([1, 2, 3, 4, 5, 6], [2, 3]);
    const swapped = swapaxes(original, 0, 1);
    assert.deepEqual(swapped.shape, [3, 2]);
    assert.deepEqual(toArray(swapped), [1, 3, 5, 2, 4, 6]);
  });

  it('add performs elementwise addition', () => {
    const a = array([1, 2, 3], [3, 1]);
    const b = array([4, 5, 6], [3, 1]);
    const result = add(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [5, 7, 9]);
  });

  it('multiply performs elementwise product', () => {
    const a = array([1, 2, 3], [3, 1]);
    const b = array([4, 5, 6], [3, 1]);
    const result = multiply(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [4, 10, 18]);
  });

  it('where selects values elementwise', () => {
    const condition = array([1, 0, 1, 0], [4, 1]);
    const x = array([10, 20, 30, 40], [4, 1]);
    const y = array([100, 200, 300, 400], [4, 1]);
    const result = where(condition, x, y);
    assert.deepEqual(result.shape, [4, 1]);
    assert.deepEqual(toArray(result), [10, 200, 30, 400]);
  });

  it('operations respect explicit streams', async () => {
    const stream = newStream();
    await withStream(stream, () => {
      const a = array([1, 2, 3, 4], [2, 2]);
      const reshaped = reshape(a, [4, 1]);
      assert.deepEqual(reshaped.shape, [4, 1]);
      const transposed = transpose(reshaped);
      assert.deepEqual(transposed.shape, [1, 4]);
    });
  });
});

describe('arange', () => {
  it('generates range from 0 to stop with single argument', () => {
    const result = arange(5);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [0, 1, 2, 3, 4]);
  });

  it('generates range from start to stop with two arguments', () => {
    const result = arange(2, 7);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [2, 3, 4, 5, 6]);
  });

  it('generates range with custom step', () => {
    const result = arange(0, 10, 2);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [0, 2, 4, 6, 8]);
  });

  it('generates range with fractional step', () => {
    const result = arange(0, 3, 0.5);
    assert.deepEqual(result.shape, [6]);
    const values = toArray(result);
    assert.equal(values.length, 6);
    assert.equal(values[0], 0);
    assert.equal(values[1], 0.5);
    assert.equal(values[2], 1);
    assert.equal(values[3], 1.5);
    assert.equal(values[4], 2);
    assert.equal(values[5], 2.5);
  });

  it('generates negative ranges with negative step', () => {
    const result = arange(0, -5, -1);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [0, -1, -2, -3, -4]);
  });

  it('returns empty array for invalid range', () => {
    const result = arange(0, -10, 1);
    assert.deepEqual(result.shape, [0]);
    assert.deepEqual(toArray(result), []);
  });

  it('handles step larger than range', () => {
    const result = arange(0, 10, 100);
    assert.deepEqual(result.shape, [1]);
    assert.deepEqual(toArray(result), [0]);
  });

  it('infers int32 dtype for integer inputs', () => {
    const result = arange(10);
    assert.equal(result.dtype, 'int32');
  });

  it('infers float32 dtype for float inputs', () => {
    const result = arange(10.0);
    assert.equal(result.dtype, 'float32');
  });

  it('respects explicit dtype parameter', () => {
    const result = arange(10, undefined, undefined, { dtype: float32 });
    assert.equal(result.dtype, 'float32');
  });

  it('works with explicit float16 dtype', () => {
    const result = arange(5, undefined, undefined, { dtype: float16 });
    assert.equal(result.dtype, 'float16');
    assert.deepEqual(result.shape, [5]);
  });

  it('works with explicit uint32 dtype', () => {
    const result = arange(5, undefined, undefined, { dtype: uint32 });
    assert.equal(result.dtype, 'uint32');
    assert.deepEqual(result.shape, [5]);
  });

  it('works with explicit int64 dtype', () => {
    const result = arange(5, undefined, undefined, { dtype: int64 });
    assert.equal(result.dtype, 'int64');
    assert.deepEqual(result.shape, [5]);
  });

  it('handles start, stop, and dtype', () => {
    const result = arange(5, 10, undefined, { dtype: float32 });
    assert.equal(result.dtype, 'float32');
    assert.deepEqual(result.shape, [5]);
  });

  it('handles start, stop, step, and dtype', () => {
    const result = arange(0, 10, 2, { dtype: float32 });
    assert.equal(result.dtype, 'float32');
    assert.deepEqual(result.shape, [5]);
    const values = toArray(result);
    assert.equal(values[0], 0);
    assert.equal(values[1], 2);
    assert.equal(values[2], 4);
  });

  it('respects explicit streams', async () => {
    const stream = newStream();
    await withStream(stream, () => {
      const result = arange(5, undefined, undefined, { stream });
      assert.deepEqual(result.shape, [5]);
      assert.deepEqual(toArray(result), [0, 1, 2, 3, 4]);
    });
  });
});
