import addon from '../internal/addon';
import MLXArray, { normalizeShapeInput } from './array';
import type { StreamLike } from './stream';
import { toNativeStreamArgument } from './stream';

function toNativeHandle(tensor: MLXArray): any {
  return tensor.toNative();
}

function normalizeStream(stream?: StreamLike | null): any {
  if (stream == null) {
    return undefined;
  }
  return toNativeStreamArgument(stream);
}

function normalizeAxes(axes: readonly number[]): number[] {
  return Array.from(axes, (axis) => {
    if (!Number.isInteger(axis)) {
      throw new Error('Axis indices must be integers');
    }
    return Number(axis);
  });
}

function normalizeAxisSpec(value: number | readonly number[], name: string): number[] {
  if (Array.isArray(value)) {
    return normalizeAxes(value);
  }
  if (!Number.isInteger(value)) {
    throw new Error(`${name} must be an integer or array of integers`);
  }
  return [Number(value)];
}

function appendStreamArg(args: any[], stream?: StreamLike | null): void {
  const native = normalizeStream(stream);
  if (native !== undefined) {
    args.push(native);
  }
}

export interface StreamOptions {
  stream?: StreamLike;
}

export interface ReshapeOptions extends StreamOptions {}

export function reshape(
  tensor: MLXArray,
  shape: readonly number[],
  options?: ReshapeOptions,
): MLXArray {
  const normalizedShape = normalizeShapeInput(shape);
  const args: any[] = [toNativeHandle(tensor), normalizedShape];
  appendStreamArg(args, options?.stream);
  const handle = addon.reshape(...args);
  return MLXArray.fromHandle(handle);
}

export interface TransposeOptions extends StreamOptions {}

export function transpose(
  tensor: MLXArray,
  axesOrOptions?: readonly number[] | TransposeOptions | null,
  maybeOptions?: TransposeOptions,
): MLXArray {
  let axes: number[] | undefined;
  let options: TransposeOptions | undefined;

  if (Array.isArray(axesOrOptions)) {
    axes = normalizeAxes(axesOrOptions);
    options = maybeOptions ?? undefined;
  } else if (axesOrOptions && 'stream' in axesOrOptions) {
    options = axesOrOptions;
  } else if (axesOrOptions != null) {
    throw new Error('transpose axes must be an array of integers');
  } else {
    options = maybeOptions;
  }

  const args: any[] = [toNativeHandle(tensor)];
  if (axes) {
    args.push(axes);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.transpose(...args);
  return MLXArray.fromHandle(handle);
}

export interface MoveAxisOptions extends StreamOptions {}

export function moveaxis(
  tensor: MLXArray,
  source: number | readonly number[],
  destination: number | readonly number[],
  options?: MoveAxisOptions,
): MLXArray {
  const src = normalizeAxisSpec(source, 'source axes');
  const dst = normalizeAxisSpec(destination, 'destination axes');
  const args: any[] = [toNativeHandle(tensor), src, dst];
  appendStreamArg(args, options?.stream);
  const handle = addon.moveaxis(...args);
  return MLXArray.fromHandle(handle);
}

export interface SwapAxesOptions extends StreamOptions {}

export function swapaxes(
  tensor: MLXArray,
  axis1: number,
  axis2: number,
  options?: SwapAxesOptions,
): MLXArray {
  if (!Number.isInteger(axis1) || !Number.isInteger(axis2)) {
    throw new Error('swapaxes expects integer axis indices');
  }
  const args: any[] = [
    toNativeHandle(tensor),
    Number(axis1),
    Number(axis2),
  ];
  appendStreamArg(args, options?.stream);
  const handle = addon.swapaxes(...args);
  return MLXArray.fromHandle(handle);
}

export interface BinaryOpOptions extends StreamOptions {}

type ScalarOrArray = MLXArray | number | boolean | bigint;

function toNativeScalarOrArray(value: ScalarOrArray): any {
  if (value instanceof MLXArray) {
    return toNativeHandle(value);
  }
  return value;
}

function binaryOp(
  name: 'add' | 'multiply',
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon[name](...args);
  return MLXArray.fromHandle(handle);
}

export function add(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  return binaryOp('add', a, b, options);
}

export function multiply(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  return binaryOp('multiply', a, b, options);
}

export interface WhereOptions extends StreamOptions {}

export function where(
  condition: MLXArray,
  onTrue: MLXArray,
  onFalse: MLXArray,
  options?: WhereOptions,
): MLXArray {
  const args: any[] = [
    toNativeHandle(condition),
    toNativeHandle(onTrue),
    toNativeHandle(onFalse),
  ];
  appendStreamArg(args, options?.stream);
  const handle = addon.where(...args);
  return MLXArray.fromHandle(handle);
}

export interface UnaryOpOptions extends StreamOptions {}

export function tan(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.tan(...args);
  return MLXArray.fromHandle(handle);
}

export interface ArangeOptions extends StreamOptions {
  dtype?: any;
}

/**
 * Generate evenly spaced values within a given interval.
 * 
 * @param start - The start of the interval (inclusive). If only one argument is provided, this argument is treated as `stop` and the interval starts at 0.
 * @param stop - The end of the interval (exclusive). Required if both `start` and `stop` are specified.
 * @param step - The spacing between values. Default is 1.
 * @param options - Optional parameters, including `dtype` and `stream`.
 * @returns An MLXArray of evenly spaced values.
 * 
 * @example
 * // Generate 0 to 9
 * arange(10)
 * 
 * @example
 * // Generate 5 to 9
 * arange(5, 10)
 * 
 * @example
 * // Generate 0 to 9 with step 2
 * arange(0, 10, 2)
 * 
 * @example
 * // Generate 0.0 to 9.5 with step 0.5
 * arange(0, 10, 0.5)
 */
export function arange(
  start: number,
  stop?: number,
  step?: number,
  options?: ArangeOptions,
): MLXArray {
  const args: any[] = [];
  
  // Handle overloaded signatures
  if (stop === undefined) {
    // Single argument: arange(stop)
    args.push(start);
  } else {
    // Two or three arguments: arange(start, stop[, step])
    args.push(start, stop);
    if (step !== undefined) {
      args.push(step);
    }
  }
  
  // Add optional dtype
  if (options?.dtype !== undefined) {
    args.push(options.dtype);
  }
  
  // Add optional stream
  appendStreamArg(args, options?.stream);
  
  const handle = addon.arange(...args);
  return MLXArray.fromHandle(handle);
}
