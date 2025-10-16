import addon from './internal/addon';
import * as utils from './utils';
import * as core from './core';
import * as streaming from './streaming';
import * as react from './react';
import * as optimizers from './optimizers';

export const native = {
  hello: (): string => addon.hello(),
};

export { core, utils, streaming, react, optimizers };
export const array = core.array;
export const Array = core.Array;
export const Stream = core.Stream;
export const issubdtype = core.issubdtype;
export const zeros = core.zeros;
export const zeros_like = core.zeros_like;
export const ones = core.ones;
export const ones_like = core.ones_like;
export const full = core.full;
export const defaultStream = core.defaultStream;
export const newStream = core.newStream;
export const setDefaultStream = core.setDefaultStream;
export const synchronize = core.synchronize;
export const streamContext = core.streamContext;
export const stream = core.stream;
export const withStream = core.withStream;
export const device = core.device;
export const reshape = core.reshape;
export const transpose = core.transpose;
export const moveaxis = core.moveaxis;
export const swapaxes = core.swapaxes;
export const add = core.add;
export const multiply = core.multiply;
export const where = core.where;
export const arange = core.arange;

// Export dtype constants
export const bool = core.bool;
export const int8 = core.int8;
export const int16 = core.int16;
export const int32 = core.int32;
export const int64 = core.int64;
export const uint8 = core.uint8;
export const uint16 = core.uint16;
export const uint32 = core.uint32;
export const uint64 = core.uint64;
export const float16 = core.float16;
export const bfloat16 = core.bfloat16;
export const float32 = core.float32;
export const float64 = core.float64;
export const complex64 = core.complex64;

export default {
  native,
  core,
  utils,
  react,
  streaming,
  optimizers,
  device,
  array,
  Array,
  issubdtype,
  zeros,
  zeros_like,
  ones,
  ones_like,
  full,
  arange,
  Stream,
  defaultStream,
  newStream,
  setDefaultStream,
  synchronize,
  streamContext,
  stream,
  withStream,
  reshape,
  transpose,
  moveaxis,
  swapaxes,
  add,
  multiply,
  where,
  bool,
  int8,
  int16,
  int32,
  int64,
  uint8,
  uint16,
  uint32,
  uint64,
  float16,
  bfloat16,
  float32,
  float64,
  complex64,
};
