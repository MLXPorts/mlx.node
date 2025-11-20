.. _nodejs_ops:

Operations
==========

MLX for Node.js provides a comprehensive set of operations for array manipulation and mathematical computations.

Element-wise Operations
-----------------------

Arithmetic Operations
^^^^^^^^^^^^^^^^^^^^^

**add(a, b)**
  Element-wise addition.

  .. code-block:: typescript

     import { array, add } from 'mlx.node';

     const a = array([1, 2, 3], [3], 'float32');
     const b = array([4, 5, 6], [3], 'float32');
     const c = add(a, b);  // [5, 7, 9]

**subtract(a, b)**
  Element-wise subtraction.

**multiply(a, b)**
  Element-wise multiplication.

**divide(a, b)**
  Element-wise division.

**power(a, b)**
  Element-wise exponentiation (a raised to power b).

Comparison Operations
^^^^^^^^^^^^^^^^^^^^^

**equal(a, b)**
  Element-wise equality comparison.

**notEqual(a, b)**
  Element-wise inequality comparison.

**less(a, b)**
  Element-wise less-than comparison.

**lessEqual(a, b)**
  Element-wise less-than-or-equal comparison.

**greater(a, b)**
  Element-wise greater-than comparison.

**greaterEqual(a, b)**
  Element-wise greater-than-or-equal comparison.

**maximum(a, b)**
  Element-wise maximum.

**minimum(a, b)**
  Element-wise minimum.

Mathematical Functions
----------------------

Trigonometric Functions
^^^^^^^^^^^^^^^^^^^^^^^

**sin(a)**
  Element-wise sine.

  .. code-block:: typescript

     import { array, sin } from 'mlx.node';

     const a = array([0, Math.PI/2, Math.PI], [3], 'float32');
     const result = sin(a);

**cos(a)**
  Element-wise cosine.

**tan(a)**
  Element-wise tangent.

**arcsin(a)**
  Element-wise inverse sine (arcsine).

**arccos(a)**
  Element-wise inverse cosine (arccosine).

**arctan(a)**
  Element-wise inverse tangent (arctangent).

**arctan2(a, b)**
  Element-wise two-argument inverse tangent.

Exponential and Logarithmic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**exp(a)**
  Element-wise exponential (e^x).

**log(a)**
  Element-wise natural logarithm.

**sqrt(a)**
  Element-wise square root.

**rsqrt(a)**
  Element-wise reciprocal square root (1/sqrt(x)).

**square(a)**
  Element-wise square (x^2).

Other Mathematical Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**abs(a)**
  Element-wise absolute value.

**sign(a)**
  Element-wise sign function.

Shape Manipulation
------------------

**reshape(a, shape)**
  Reshape an array without changing its data.

  :param a: Input array
  :param shape: New shape as array of numbers
  :returns: Reshaped array

  .. code-block:: typescript

     import { array, reshape } from 'mlx.node';

     const a = array([1, 2, 3, 4, 5, 6], [6], 'float32');
     const b = reshape(a, [2, 3]);  // [[1, 2, 3], [4, 5, 6]]

**transpose(a, axes)**
  Transpose the dimensions of an array.

  :param a: Input array
  :param axes: Optional permutation of axes
  :returns: Transposed array

  .. code-block:: typescript

     import { array, transpose } from 'mlx.node';

     const a = array([1, 2, 3, 4], [2, 2], 'float32');
     const b = transpose(a);  // [[1, 3], [2, 4]]

**moveaxis(a, source, destination)**
  Move axes of an array to new positions.

**swapaxes(a, axis1, axis2)**
  Swap two axes of an array.

Array Generation
----------------

**arange(start, stop, step, dtype)**
  Create an array with evenly spaced values.

  :param start: Start value
  :param stop: Stop value (exclusive)
  :param step: Step size (default: 1)
  :param dtype: Data type (default: inferred)
  :returns: Array of evenly spaced values

  .. code-block:: typescript

     import { arange } from 'mlx.node';

     const a = arange(0, 10, 2, 'int32');  // [0, 2, 4, 6, 8]

Conditional Operations
----------------------

**where(condition, x, y)**
  Select elements from x or y depending on condition.

  :param condition: Boolean array for condition
  :param x: Array to select from where condition is true
  :param y: Array to select from where condition is false
  :returns: Array with selected elements

  .. code-block:: typescript

     import { array, where, greater } from 'mlx.node';

     const a = array([1, 2, 3, 4], [4], 'float32');
     const threshold = array([2.5], [1], 'float32');
     const result = where(greater(a, threshold), a, threshold);

Naming Conventions
------------------

Note that MLX for Node.js uses TypeScript/JavaScript naming conventions (camelCase)
rather than Python's snake_case:

- ``zeros_like`` → ``zerosLike``
- ``ones_like`` → ``onesLike``
- ``not_equal`` → ``notEqual``
- ``less_equal`` → ``lessEqual``
- ``greater_equal`` → ``greaterEqual``

Usage Patterns
--------------

Named Exports (Idiomatic TypeScript)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { array, add, multiply, sin } from 'mlx.node';

   const a = array([1, 2, 3], [3], 'float32');
   const b = add(a, a);
   const c = sin(multiply(b, a));

Core Namespace (Python-like)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { core } from 'mlx.node';

   const a = core.array([1, 2, 3], [3], 'float32');
   const b = core.add(a, a);
   const c = core.sin(core.multiply(b, a));

See Also
--------

- :ref:`nodejs_array` - Array creation and manipulation
- :ref:`nodejs_data_types` - Supported data types
- :ref:`nodejs_quickstart` - Getting started guide
