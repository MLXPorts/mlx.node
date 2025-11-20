.. _nodejs_array:

Array
=====

The ``Array`` class is the core data structure in MLX for Node.js. It represents
a multi-dimensional array that can be operated on using the Metal GPU.

Creating Arrays
---------------

Factory Functions
^^^^^^^^^^^^^^^^^

**array(data, shape, dtype)**
  Create an array from data.

  :param data: TypedArray, Array, or buffer containing the data
  :param shape: Array of numbers defining the shape
  :param dtype: Data type (e.g., 'float32', 'int32')
  :returns: MLXArray instance

  .. code-block:: typescript

     import { array } from 'mlx.node';

     const arr = array([1, 2, 3, 4], [2, 2], 'float32');

**zeros(shape, dtype)**
  Create an array filled with zeros.

  :param shape: Array of numbers defining the shape
  :param dtype: Data type (default: 'float32')
  :returns: MLXArray instance

  .. code-block:: typescript

     import { zeros } from 'mlx.node';

     const z = zeros([3, 3], 'float32');

**ones(shape, dtype)**
  Create an array filled with ones.

  :param shape: Array of numbers defining the shape
  :param dtype: Data type (default: 'float32')
  :returns: MLXArray instance

  .. code-block:: typescript

     import { ones } from 'mlx.node';

     const o = ones([2, 4], 'int32');

**full(shape, value, dtype)**
  Create an array filled with a specific value.

  :param shape: Array of numbers defining the shape
  :param value: Scalar value or MLXArray to fill with
  :param dtype: Data type (inferred from value if not specified)
  :returns: MLXArray instance

  .. code-block:: typescript

     import { full } from 'mlx.node';

     const f = full([3, 3], 7.5, 'float32');

**zerosLike(array)**
  Create an array of zeros with the same shape and dtype as the input.

  :param array: Reference array
  :returns: MLXArray instance

**onesLike(array)**
  Create an array of ones with the same shape and dtype as the input.

  :param array: Reference array
  :returns: MLXArray instance

Array Properties
----------------

**shape**
  Array of numbers representing the dimensions of the array.

  .. code-block:: typescript

     const arr = array([1, 2, 3, 4], [2, 2], 'float32');
     console.log(arr.shape);  // [2, 2]

**dtype**
  String representing the data type of the array elements.

  .. code-block:: typescript

     const arr = array([1, 2, 3], [3], 'float32');
     console.log(arr.dtype);  // 'float32'

**ndim**
  Number of dimensions (rank) of the array.

  .. code-block:: typescript

     const arr = array([1, 2, 3, 4], [2, 2], 'float32');
     console.log(arr.ndim);  // 2

**size**
  Total number of elements in the array.

  .. code-block:: typescript

     const arr = array([1, 2, 3, 4], [2, 2], 'float32');
     console.log(arr.size);  // 4

Array Methods
-------------

**toArray()**
  Convert the MLX array to a JavaScript array (nested for multi-dimensional).

  :returns: JavaScript Array

  .. code-block:: typescript

     const arr = array([1, 2, 3, 4], [2, 2], 'float32');
     const jsArr = arr.toArray();  // [[1, 2], [3, 4]]

**toFloat32Array()**
  Convert to a Float32Array (flattened).

  :returns: Float32Array

  .. code-block:: typescript

     const arr = array([1, 2, 3], [3], 'float32');
     const typed = arr.toFloat32Array();  // Float32Array [1, 2, 3]

**toTypedArray()**
  Convert to the appropriate TypedArray based on dtype.

  :returns: TypedArray (Float32Array, Int32Array, etc.)

  .. code-block:: typescript

     const intArr = array([1, 2, 3], [3], 'int32');
     const typed = intArr.toTypedArray();  // Int32Array [1, 2, 3]

Static Methods
--------------

**Array.from(data, shape, dtype)**
  Alternative way to create an array using the class.

  .. code-block:: typescript

     import { Array as MLXArray } from 'mlx.node';

     const arr = MLXArray.from([1, 2, 3], [3], 'float32');

Type Annotations
----------------

When using TypeScript, you can import and use the Array type:

.. code-block:: typescript

   import { Array as MLXArray, array } from 'mlx.node';

   const arr: MLXArray = array([1, 2, 3], [3], 'float32');

   function processArray(arr: MLXArray): MLXArray {
     return add(arr, ones(arr.shape, arr.dtype));
   }

See Also
--------

- :ref:`nodejs_ops` - Array operations
- :ref:`nodejs_data_types` - Supported data types
- :ref:`nodejs_quickstart` - Getting started guide
