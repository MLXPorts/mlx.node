.. _nodejs_quickstart:

Node.js Quickstart
==================

This guide will get you started with MLX in Node.js.

Installation
------------

Install MLX for Node.js using npm:

.. code-block:: shell

   npm install mlx.node

Basic Usage
-----------

Import MLX and create arrays:

.. code-block:: typescript

   import { array, zeros, ones } from 'mlx.node';

   // Create an array from data
   const a = array([1, 2, 3, 4], [2, 2], 'float32');
   console.log(a.toArray());  // [[1, 2], [3, 4]]

   // Create arrays filled with zeros or ones
   const z = zeros([3, 3], 'float32');
   const o = ones([2, 4], 'int32');

Array Operations
----------------

Perform operations on arrays:

.. code-block:: typescript

   import { array, add, multiply, sin, cos } from 'mlx.node';

   const a = array([1.0, 2.0, 3.0], [3], 'float32');
   const b = array([4.0, 5.0, 6.0], [3], 'float32');

   // Element-wise operations
   const c = add(a, b);           // [5, 7, 9]
   const d = multiply(a, b);      // [4, 10, 18]

   // Mathematical functions
   const e = sin(a);
   const f = cos(a);

Using Core Namespace
--------------------

For Python-like API, use the core namespace:

.. code-block:: typescript

   import { core } from 'mlx.node';

   const arr = core.array([1, 2, 3], [3], 'float32');
   const result = core.add(arr, core.ones([3], 'float32'));

Working with Different Data Types
----------------------------------

MLX supports various data types:

.. code-block:: typescript

   import { array, float32, int32, bool } from 'mlx.node';

   const floatArr = array([1.5, 2.5, 3.5], [3], float32);
   const intArr = array([1, 2, 3], [3], int32);
   const boolArr = array([1, 0, 1], [3], bool);

   // Access dtype information
   console.log(floatArr.dtype);  // 'float32'
   console.log(float32.key);     // 'float32'
   console.log(float32.size);    // 4 bytes

Shape and Reshape
-----------------

Work with array shapes:

.. code-block:: typescript

   import { array, reshape } from 'mlx.node';

   const a = array([1, 2, 3, 4, 5, 6], [6], 'float32');
   const b = reshape(a, [2, 3]);  // [[1, 2, 3], [4, 5, 6]]
   const c = reshape(a, [3, 2]);  // [[1, 2], [3, 4], [5, 6]]

   console.log(a.shape);  // [6]
   console.log(b.shape);  // [2, 3]

Lazy Evaluation
---------------

MLX uses lazy evaluation. Arrays are only materialized when needed:

.. code-block:: typescript

   import { array, add, multiply } from 'mlx.node';

   const a = array([1, 2, 3], [3], 'float32');
   const b = array([4, 5, 6], [3], 'float32');

   // These operations build a computation graph
   const c = add(a, b);
   const d = multiply(c, a);

   // Array is materialized when you access the data
   const result = d.toArray();  // Computation happens here

TypeScript Support
------------------

MLX for Node.js has full TypeScript support with proper type definitions:

.. code-block:: typescript

   import { Array as MLXArray, array } from 'mlx.node';

   // Type annotations
   const arr: MLXArray = array([1, 2, 3], [3], 'float32');

   // TypeScript infers types
   const zeros = core.zeros([2, 2], 'float32');  // Type: MLXArray

Next Steps
----------

- Explore the :ref:`nodejs_array` API reference
- Learn about available :ref:`nodejs_ops` operations
- Understand :ref:`nodejs_data_types` in MLX
- Check out :ref:`nodejs_examples` for more complex use cases
