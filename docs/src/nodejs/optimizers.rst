.. _nodejs_optimizers:

Optimizers
==========

MLX for Node.js provides optimizer implementations for training neural networks.
These optimizers update model parameters based on computed gradients.

Available Optimizers
--------------------

SGD (Stochastic Gradient Descent)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basic gradient descent with optional momentum.

.. code-block:: typescript

   import { optimizers } from 'mlx.node';

   const optimizer = new optimizers.SGD({
     learningRate: 0.01,
     momentum: 0.9
   });

**Parameters:**

- ``learningRate`` (number): Learning rate for parameter updates (default: 0.01)
- ``momentum`` (number): Momentum factor (default: 0)

Adam
^^^^

Adaptive Moment Estimation optimizer.

.. code-block:: typescript

   import { optimizers } from 'mlx.node';

   const optimizer = new optimizers.Adam({
     learningRate: 0.001,
     betas: [0.9, 0.999],
     eps: 1e-8
   });

**Parameters:**

- ``learningRate`` (number): Learning rate (default: 0.001)
- ``betas`` (array): Coefficients for computing running averages [beta1, beta2] (default: [0.9, 0.999])
- ``eps`` (number): Small constant for numerical stability (default: 1e-8)

AdamW
^^^^^

Adam optimizer with decoupled weight decay.

.. code-block:: typescript

   import { optimizers } from 'mlx.node';

   const optimizer = new optimizers.AdamW({
     learningRate: 0.001,
     betas: [0.9, 0.999],
     eps: 1e-8,
     weightDecay: 0.01
   });

**Parameters:**

- ``learningRate`` (number): Learning rate (default: 0.001)
- ``betas`` (array): Coefficients for running averages (default: [0.9, 0.999])
- ``eps`` (number): Numerical stability constant (default: 1e-8)
- ``weightDecay`` (number): Weight decay coefficient (default: 0.01)

AdaGrad
^^^^^^^

Adaptive gradient optimizer with per-parameter learning rates.

.. code-block:: typescript

   import { optimizers } from 'mlx.node';

   const optimizer = new optimizers.AdaGrad({
     learningRate: 0.01,
     eps: 1e-8
   });

**Parameters:**

- ``learningRate`` (number): Learning rate (default: 0.01)
- ``eps`` (number): Numerical stability constant (default: 1e-8)

RMSprop
^^^^^^^

Root Mean Square Propagation optimizer.

.. code-block:: typescript

   import { optimizers } from 'mlx.node';

   const optimizer = new optimizers.RMSprop({
     learningRate: 0.01,
     alpha: 0.99,
     eps: 1e-8
   });

**Parameters:**

- ``learningRate`` (number): Learning rate (default: 0.01)
- ``alpha`` (number): Smoothing constant (default: 0.99)
- ``eps`` (number): Numerical stability constant (default: 1e-8)

Adafactor
^^^^^^^^^

Memory-efficient adaptive learning rate optimizer.

.. code-block:: typescript

   import { optimizers } from 'mlx.node';

   const optimizer = new optimizers.Adafactor({
     learningRate: null,  // Use adaptive learning rate
     eps: [1e-30, 1e-3],
     clipThreshold: 1.0,
     decayRate: -0.8,
     beta1: null,
     weightDecay: 0.0
   });

**Parameters:**

- ``learningRate`` (number | null): Learning rate, or null for adaptive (default: null)
- ``eps`` (array): Regularization constants [eps1, eps2] (default: [1e-30, 1e-3])
- ``clipThreshold`` (number): Threshold for gradient clipping (default: 1.0)
- ``decayRate`` (number): Decay rate for second moment (default: -0.8)
- ``beta1`` (number | null): Beta1 for momentum, or null (default: null)
- ``weightDecay`` (number): Weight decay coefficient (default: 0.0)

Lion
^^^^

EvoLved Sign Momentum optimizer.

.. code-block:: typescript

   import { optimizers } from 'mlx.node';

   const optimizer = new optimizers.Lion({
     learningRate: 0.0001,
     betas: [0.9, 0.99],
     weightDecay: 0.0
   });

**Parameters:**

- ``learningRate`` (number): Learning rate (default: 0.0001)
- ``betas`` (array): Coefficients for momentum [beta1, beta2] (default: [0.9, 0.99])
- ``weightDecay`` (number): Weight decay coefficient (default: 0.0)

Optimizer Interface
-------------------

All optimizers share a common interface:

**update(model, gradients)**
  Update model parameters using computed gradients.

  :param model: Object with parameters to update
  :param gradients: Object with gradients for each parameter
  :returns: Updated model

  .. code-block:: typescript

     import { optimizers } from 'mlx.node';

     const optimizer = new optimizers.Adam({ learningRate: 0.001 });

     // In training loop
     const gradients = computeGradients(model, loss);
     model = optimizer.update(model, gradients);

**state**
  Get or set the optimizer's internal state.

  .. code-block:: typescript

     const state = optimizer.state;
     // Save state for checkpointing
     // Later: restore state
     optimizer.state = savedState;

Usage Example
-------------

Complete Training Loop
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

   import { array, core, optimizers } from 'mlx.node';

   // Initialize model parameters
   let model = {
     weights: core.array(new Float32Array(100), [10, 10], 'float32'),
     bias: core.zeros([10], 'float32')
   };

   // Create optimizer
   const optimizer = new optimizers.Adam({
     learningRate: 0.001
   });

   // Training loop
   for (let epoch = 0; epoch < 100; epoch++) {
     // Forward pass
     const predictions = forward(model, inputs);

     // Compute loss
     const loss = computeLoss(predictions, targets);

     // Backward pass (compute gradients)
     const gradients = backward(loss);

     // Update parameters
     model = optimizer.update(model, gradients);

     console.log(`Epoch ${epoch}, Loss: ${loss}`);
   }

Choosing an Optimizer
---------------------

**SGD**: Simple and effective for many problems. Add momentum for better convergence.

**Adam**: Good default choice for most deep learning tasks. Adaptive learning rates per parameter.

**AdamW**: Preferred over Adam when using weight decay regularization.

**AdaGrad**: Good for sparse gradients and problems with infrequent features.

**RMSprop**: Works well for recurrent neural networks.

**Adafactor**: Memory-efficient for very large models.

**Lion**: Newer optimizer that can be more memory-efficient than Adam while maintaining performance.

Learning Rate Scheduling
------------------------

You can implement learning rate scheduling by updating the optimizer's learning rate:

.. code-block:: typescript

   import { optimizers } from 'mlx.node';

   const optimizer = new optimizers.Adam({ learningRate: 0.001 });

   for (let epoch = 0; epoch < 100; epoch++) {
     // Decay learning rate
     if (epoch % 10 === 0) {
       optimizer.learningRate *= 0.9;
     }

     // Training step
     // ...
   }

Best Practices
--------------

1. **Start with Adam**: It's a good default optimizer for most problems.

2. **Use AdamW with weight decay**: Prefer AdamW over Adam when applying weight decay.

3. **Tune learning rate**: Learning rate is the most important hyperparameter. Start with recommended defaults and adjust based on training behavior.

4. **Monitor gradient norms**: Large gradients may require gradient clipping or lower learning rates.

5. **Save optimizer state**: Include optimizer state in model checkpoints for resuming training.

6. **Consider memory usage**: Optimizers like Adafactor use less memory than Adam for large models.

See Also
--------

- :ref:`nodejs_array` - Array manipulation
- :ref:`nodejs_ops` - Operations for computing gradients
- :ref:`nodejs_quickstart` - Getting started guide
