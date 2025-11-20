.. _nodejs_examples:

Examples
========

This section provides practical examples of using MLX for Node.js in various
scenarios.

Basic Array Operations
----------------------

Creating and Manipulating Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { array, zeros, ones, reshape, transpose } from 'mlx.node';

   // Create arrays
   const a = array([1, 2, 3, 4, 5, 6], [6], 'float32');
   const b = zeros([3, 3], 'float32');
   const c = ones([2, 4], 'int32');

   // Reshape
   const reshaped = reshape(a, [2, 3]);
   console.log(reshaped.shape);  // [2, 3]

   // Transpose
   const transposed = transpose(reshaped);
   console.log(transposed.shape);  // [3, 2]

   // Access data
   console.log(reshaped.toArray());  // [[1, 2, 3], [4, 5, 6]]

Mathematical Computations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { array, add, multiply, sin, cos, sqrt } from 'mlx.node';

   const x = array([0, Math.PI/4, Math.PI/2], [3], 'float32');

   // Trigonometric operations
   const sinX = sin(x);
   const cosX = cos(x);

   // Verify Pythagorean identity: sin²(x) + cos²(x) = 1
   const sinSquared = multiply(sinX, sinX);
   const cosSquared = multiply(cosX, cosX);
   const identity = add(sinSquared, cosSquared);

   console.log(identity.toArray());  // [1, 1, 1] (approximately)

Simple Linear Regression
-------------------------

.. code-block:: typescript

   import { core } from 'mlx.node';

   // Generate synthetic data: y = 2x + 1 + noise
   const x = core.arange(0, 100, 1, 'float32');
   const noise = core.multiply(
     core.array(Array.from({length: 100}, () => Math.random() - 0.5), [100], 'float32'),
     core.array([0.5], [1], 'float32')
   );
   const y = core.add(
     core.add(core.multiply(core.array([2], [1], 'float32'), x), core.array([1], [1], 'float32')),
     noise
   );

   // Initialize parameters
   let w = core.zeros([1], 'float32');
   let b = core.zeros([1], 'float32');

   const learningRate = 0.001;
   const epochs = 100;

   for (let epoch = 0; epoch < epochs; epoch++) {
     // Forward pass: y_pred = w * x + b
     const yPred = core.add(core.multiply(w, x), b);

     // Compute loss: MSE
     const diff = core.subtract(yPred, y);
     const squaredDiff = core.multiply(diff, diff);
     const loss = core.divide(
       core.array([squaredDiff.toArray().reduce((a, b) => a + b, 0)], [1], 'float32'),
       core.array([x.size], [1], 'float32')
     );

     // Compute gradients
     const dw = core.divide(
       core.array([2 * diff.toArray().reduce((sum, d, i) => sum + d * x.toArray()[i], 0)], [1], 'float32'),
       core.array([x.size], [1], 'float32')
     );
     const db = core.divide(
       core.array([2 * diff.toArray().reduce((a, b) => a + b, 0)], [1], 'float32'),
       core.array([x.size], [1], 'float32')
     );

     // Update parameters
     w = core.subtract(w, core.multiply(core.array([learningRate], [1], 'float32'), dw));
     b = core.subtract(b, core.multiply(core.array([learningRate], [1], 'float32'), db));

     if (epoch % 10 === 0) {
       console.log(`Epoch ${epoch}: Loss = ${loss.toArray()[0]}, w = ${w.toArray()[0]}, b = ${b.toArray()[0]}`);
     }
   }

   console.log(`Final: w = ${w.toArray()[0]}, b = ${b.toArray()[0]}`);
   // Expected: w ≈ 2.0, b ≈ 1.0

Matrix Operations
-----------------

Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { array, core } from 'mlx.node';

   const A = array([1, 2, 3, 4], [2, 2], 'float32');
   const B = array([5, 6, 7, 8], [2, 2], 'float32');

   // Element-wise multiplication
   const elementWise = core.multiply(A, B);
   console.log(elementWise.toArray());
   // [[5, 12], [21, 32]]

   // Matrix multiplication would use matmul when implemented
   // const matMul = core.matmul(A, B);

Working with Different Data Types
----------------------------------

Type Conversions and Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { array, core, float32, int32, bool as boolType } from 'mlx.node';

   // Float array
   const floats = array([1.5, 2.7, 3.2], [3], float32);
   console.log(`Float dtype: ${floats.dtype}`);

   // Integer array
   const integers = array([1, 2, 3], [3], int32);
   console.log(`Int dtype: ${integers.dtype}`);

   // Boolean array
   const booleans = array([1, 0, 1], [3], boolType);
   console.log(`Bool values: ${booleans.toArray()}`);
   // [true, false, true]

   // Check dtype hierarchy
   console.log(core.issubdtype(float32, core.floating));  // true
   console.log(core.issubdtype(int32, core.integer));     // true
   console.log(core.issubdtype(float32, core.integer));   // false

Conditional Operations
----------------------

Using where() for Conditional Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { array, where, greater, core } from 'mlx.node';

   const data = array([1, 5, 3, 8, 2, 9], [6], 'float32');
   const threshold = array([5], [1], 'float32');

   // Clip values to threshold
   const condition = greater(data, threshold);
   const clipped = where(condition, threshold, data);

   console.log(data.toArray());     // [1, 5, 3, 8, 2, 9]
   console.log(clipped.toArray());  // [1, 5, 3, 5, 2, 5]

Using Streams
-------------

Parallel Computations
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { newStream, withStream, array, add, multiply } from 'mlx.node';

   const stream1 = newStream('gpu');
   const stream2 = newStream('gpu');

   // Computation 1 on stream 1
   const result1 = withStream(stream1, () => {
     const a = array([1, 2, 3], [3], 'float32');
     const b = array([4, 5, 6], [3], 'float32');
     return add(a, b);
   });

   // Computation 2 on stream 2 (runs concurrently)
   const result2 = withStream(stream2, () => {
     const c = array([7, 8, 9], [3], 'float32');
     const d = array([2, 2, 2], [3], 'float32');
     return multiply(c, d);
   });

   console.log(result1.toArray());  // [5, 7, 9]
   console.log(result2.toArray());  // [14, 16, 18]

Next.js Integration
-------------------

Server-Side Inference with Streaming
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   // app/api/inference/route.ts
   import { streaming, core, newStream, withStream } from 'mlx.node';

   async function* performInference() {
     const inferenceStream = newStream('gpu');

     await withStream(inferenceStream, async () => {
       // Simulate inference steps
       for (let step = 0; step < 10; step++) {
         const result = core.array([step], [1], 'float32');
         yield result;

         // Small delay to simulate computation
         await new Promise(resolve => setTimeout(resolve, 100));
       }
     });
   }

   export async function GET() {
     return streaming.eventStreamResponse(performInference());
   }

Client-Side Consumption
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: javascript

   // app/components/InferenceClient.tsx
   'use client';

   import { useEffect, useState } from 'react';

   export function InferenceClient() {
     const [results, setResults] = useState<number[]>([]);

     useEffect(() => {
       const eventSource = new EventSource('/api/inference');

       eventSource.onmessage = (event) => {
         const data = JSON.parse(event.data);
         if (data.type === 'data') {
           setResults(prev => [...prev, data.value]);
         }
       };

       return () => eventSource.close();
     }, []);

     return (
       <div>
         <h2>Inference Results:</h2>
         <ul>
           {results.map((result, i) => (
             <li key={i}>Step {i}: {result}</li>
           ))}
         </ul>
       </div>
     );
   }

Using Optimizers
----------------

Training with Adam Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { core, optimizers } from 'mlx.node';

   // Simple model parameters
   let model = {
     w1: core.array(new Float32Array(20).fill(0.1), [4, 5], 'float32'),
     b1: core.zeros([5], 'float32'),
     w2: core.array(new Float32Array(10).fill(0.1), [5, 2], 'float32'),
     b2: core.zeros([2], 'float32')
   };

   // Create optimizer
   const optimizer = new optimizers.Adam({
     learningRate: 0.001,
     betas: [0.9, 0.999]
   });

   // Training loop
   for (let epoch = 0; epoch < 100; epoch++) {
     // Compute gradients (pseudo-code - actual gradient computation needed)
     const gradients = {
       w1: computeGradient(model.w1),
       b1: computeGradient(model.b1),
       w2: computeGradient(model.w2),
       b2: computeGradient(model.b2)
     };

     // Update model
     model = optimizer.update(model, gradients);

     if (epoch % 10 === 0) {
       console.log(`Epoch ${epoch} complete`);
     }
   }

See Also
--------

- :ref:`nodejs_array` - Array API reference
- :ref:`nodejs_ops` - Operations reference
- :ref:`nodejs_optimizers` - Optimizer reference
- :ref:`nodejs_devices_and_streams` - Streams and devices
- :ref:`node_streaming` - Streaming architecture details
