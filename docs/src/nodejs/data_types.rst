.. _nodejs_data_types:

Data Types
==========

MLX for Node.js supports the same data types as MLX Python. The default floating
point type is ``float32`` and the default integer type is ``int32``.

Supported Data Types
--------------------

.. list-table:: Data Types in MLX
   :widths: 5 3 20
   :header-rows: 1

   * - Type
     - Bytes
     - Description
   * - ``bool``
     - 1
     - Boolean (true, false) data type
   * - ``uint8``
     - 1
     - 8-bit unsigned integer
   * - ``uint16``
     - 2
     - 16-bit unsigned integer
   * - ``uint32``
     - 4
     - 32-bit unsigned integer
   * - ``uint64``
     - 8
     - 64-bit unsigned integer
   * - ``int8``
     - 1
     - 8-bit signed integer
   * - ``int16``
     - 2
     - 16-bit signed integer
   * - ``int32``
     - 4
     - 32-bit signed integer
   * - ``int64``
     - 8
     - 64-bit signed integer
   * - ``bfloat16``
     - 2
     - 16-bit brain float (e8, m7)
   * - ``float16``
     - 2
     - 16-bit IEEE float (e5, m10)
   * - ``float32``
     - 4
     - 32-bit float
   * - ``float64``
     - 8
     - 64-bit double
   * - ``complex64``
     - 8
     - 64-bit complex float

.. note::

    Arrays with type ``float64`` only work with CPU operations. Using
    ``float64`` arrays on the GPU will result in an exception.

Using Data Types
----------------

As String Literals
^^^^^^^^^^^^^^^^^^

The simplest way to specify a data type is using string literals:

.. code-block:: typescript

   import { array, zeros } from 'mlx.node';

   const floatArr = array([1, 2, 3], [3], 'float32');
   const intArr = array([1, 2, 3], [3], 'int32');
   const boolArr = array([1, 0, 1], [3], 'bool');

As Dtype Constants
^^^^^^^^^^^^^^^^^^

You can also use the exported dtype constants for better type safety:

.. code-block:: typescript

   import { array, float32, int32, bool as boolType } from 'mlx.node';

   const floatArr = array([1, 2, 3], [3], float32);
   const intArr = array([1, 2, 3], [3], int32);
   const boolArr = array([1, 0, 1], [3], boolType);

Dtype Objects
-------------

Each dtype is represented by a Dtype object with properties:

.. code-block:: typescript

   import { core } from 'mlx.node';

   const f32 = core.float32;
   console.log(f32.key);       // 'float32'
   console.log(f32.size);      // 4 (bytes)
   console.log(f32.category);  // 'floating'

Dtype Methods
^^^^^^^^^^^^^

**equals(other)**
  Check if two dtypes are equal.

  .. code-block:: typescript

     import { core } from 'mlx.node';

     const isEqual = core.float32.equals(core.float32);  // true

**toString()**
  Get string representation of the dtype.

  .. code-block:: typescript

     import { core } from 'mlx.node';

     console.log(core.float32.toString());  // 'core.float32'

Dtype Utilities
---------------

**core.dtype.fromString(key)**
  Get a Dtype object from its string key.

  .. code-block:: typescript

     import { core } from 'mlx.node';

     const dtype = core.dtype.fromString('int32');
     console.log(dtype.key);  // 'int32'

**core.dtype.keys()**
  Get all dtype keys.

  .. code-block:: typescript

     import { core } from 'mlx.node';

     const keys = core.dtype.keys();
     // ['bool', 'uint8', 'uint16', ..., 'complex64']

**core.dtype.values()**
  Get all Dtype objects.

**core.dtype.items()**
  Get all dtype key-value pairs.

**core.dtype.has(key)**
  Check if a dtype key exists.

**core.dtype.get(key)**
  Get a Dtype object by key (same as fromString).

Dtype Categories
----------------

Data types are organized in a hierarchy. Use dtype categories for type checking:

.. code-block:: typescript

   import { core } from 'mlx.node';

   const floating = core.floating;
   console.log(floating.name);  // 'floating'

   const integer = core.integer;
   const number = core.number;
   const generic = core.generic;

Category Methods
^^^^^^^^^^^^^^^^

**core.dtype.categoryKeys()**
  Get all category names.

  .. code-block:: typescript

     import { core } from 'mlx.node';

     const categories = core.dtype.categoryKeys();
     // ['generic', 'number', 'integer', 'floating', ...]

**core.dtype.categoryValues()**
  Get all DtypeCategory objects.

**core.dtype.categoryItems()**
  Get all category key-value pairs.

Type Hierarchy Checks
---------------------

**issubdtype(dtype1, dtype2)**
  Check if dtype1 is a subtype of dtype2 (or category).

  .. code-block:: typescript

     import { core } from 'mlx.node';

     // Check if float32 is a floating type
     core.issubdtype(core.float32, core.floating);  // true

     // Check if float32 is a number
     core.issubdtype(core.float32, core.number);  // true

     // Check if float32 is an integer
     core.issubdtype(core.float32, core.integer);  // false

     // Check exact type
     core.issubdtype(core.float32, core.float32);  // true

Type Conversions
----------------

Arrays can be created from TypedArrays, which automatically determine the dtype:

.. code-block:: typescript

   import { array } from 'mlx.node';

   const float32Data = new Float32Array([1, 2, 3]);
   const int32Data = new Int32Array([1, 2, 3]);
   const uint8Data = new Uint8Array([1, 0, 1]);

   const floatArr = array(float32Data, [3], 'float32');
   const intArr = array(int32Data, [3], 'int32');
   const boolArr = array(uint8Data, [3], 'bool');

Converting Back to TypedArrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { array } from 'mlx.node';

   const mlxArr = array([1, 2, 3], [3], 'float32');

   // Get specific typed array
   const float32 = mlxArr.toFloat32Array();  // Float32Array

   // Get appropriate typed array based on dtype
   const typed = mlxArr.toTypedArray();  // Float32Array

   // Get nested JavaScript array
   const jsArr = mlxArr.toArray();  // [1, 2, 3]

Best Practices
--------------

1. **Use float32 for GPU operations**: It's the default and most efficient on Apple Silicon.

2. **Avoid float64 on GPU**: float64 only works on CPU and will throw if used with GPU operations.

3. **Use constants for type safety**: Import dtype constants instead of string literals when possible.

4. **Check types with issubdtype**: Use dtype hierarchy checks for flexible type validation.

See Also
--------

- :ref:`nodejs_array` - Array creation and manipulation
- :ref:`nodejs_ops` - Array operations
- :ref:`nodejs_quickstart` - Getting started guide
