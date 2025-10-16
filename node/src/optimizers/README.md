# MLX Optimizers for Node.js

This module provides optimizer implementations for training neural networks with MLX in Node.js.

## Implementation Status

### ✅ Completed

- **Base `Optimizer` class**: Abstract base class with state management, scheduler support, and parameter tree handling
- **`SGD` optimizer class**: Stochastic Gradient Descent with momentum, weight decay, dampening, and Nesterov momentum support
- **`Lion` optimizer class**: Lion (EvoLved Sign Momentum) optimizer with momentum and weight decay support
- **Core operations**: Added `sign` and `subtract` operations for optimizer support
- **API structure**: Matches Python MLX API for optimizer initialization and configuration
- **Type safety**: Full TypeScript types and interfaces
- **Validation**: Parameter validation (e.g., Nesterov requirements)
- **State management**: Proper state initialization and tracking
- **Tests**: Basic unit tests for constructor, validation, and state management

### ⚠️ Partially Implemented

The `SGD.applySingle()` method has a placeholder implementation because it requires some additional operations:

**Available Core Operations:**
- ✅ `subtract(a, b)` - Array subtraction (now available)
- ✅ `sign(a)` - Sign function (now available)
- ✅ Scalar-array arithmetic via `add`, `multiply`, `subtract` with scalar support
- ⚠️ `astype()` - Dtype conversion (used internally but not exposed as method)
- ⚠️ Proper scalar array construction from numbers

**Current Workarounds:**
- Step counter uses `zeros([])` instead of proper scalar construction
- Learning rate setting creates empty arrays instead of proper scalar values
- `SGD.applySingle()` throws an error explaining the missing dependencies
- `Lion.applySingle()` is fully implemented and working

## Usage Example

```typescript
import { SGD } from 'mlx/optimizers';
import { zeros } from 'mlx/core';

// Create optimizer
const optimizer = new SGD({
  learningRate: 0.01,
  momentum: 0.9,
  weightDecay: 0.0001,
  nesterov: false
});

// Initialize with parameters
const parameters = {
  weight: zeros([10, 5]),
  bias: zeros([5])
};
optimizer.init(parameters);

// During training (once core operations are available):
// const updatedParams = optimizer.applyGradients(gradients, parameters);
```

## API Reference

### `Optimizer` (Abstract Base Class)

**Properties:**
- `state: Record<string, any>` - The optimizer's state dictionary
- `step: MLXArray` - Current step count
- `learningRate: MLXArray` - Current learning rate

**Methods:**
- `init(parameters)` - Initialize optimizer state for a parameter tree
- `applyGradients(gradients, parameters)` - Apply gradients and return updated parameters
- `initSingle(parameter, state)` - Initialize state for a single parameter (abstract)
- `applySingle(gradient, parameter, state)` - Apply update to a single parameter (abstract)

### `SGD` (Stochastic Gradient Descent)

**Constructor Options:**
```typescript
interface SGDOptions {
  learningRate: number | Scheduler;  // Learning rate (can be scheduled)
  momentum?: number;                 // Momentum strength (default: 0)
  weightDecay?: number;              // L2 penalty (default: 0)
  dampening?: number;                // Dampening for momentum (default: 0)
  nesterov?: boolean;                // Enable Nesterov momentum (default: false)
}
```

**Update Formula:**
```
v_{t+1} = μ * v_t + (1 - τ) * g_t
w_{t+1} = w_t - λ * v_{t+1}
```

Where:
- `λ` is the learning rate
- `μ` is the momentum strength
- `τ` is the dampening
- For Nesterov momentum: use `g_t + μ * v_{t+1}` instead of `v_{t+1}`

**Validation:**
- Nesterov momentum requires `momentum > 0` and `dampening = 0`
- Throws error if validation fails

### `Lion` (EvoLved Sign Momentum)

**Constructor Options:**
```typescript
interface LionOptions {
  learningRate: number | Scheduler;  // Learning rate (can be scheduled)
  betas?: [number, number];          // Beta coefficients (default: [0.9, 0.99])
  weightDecay?: number;              // Weight decay (default: 0.0)
}
```

**Update Formula:**
```
c_{t+1} = β₁ * m_t + (1 - β₁) * g_t
m_{t+1} = β₂ * m_t + (1 - β₂) * g_t
w_{t+1} = w_t - η * (sign(c_t) + λ * w_t)
```

Where:
- `η` is the learning rate
- `β₁` and `β₂` are the beta coefficients
- `λ` is the weight decay

**Notes:**
- Updates use the sign operation, resulting in larger norm updates than SGD/Adam
- Recommended: learning rate 3-10x smaller than AdamW
- Recommended: weight decay 3-10x larger than AdamW
- Reference: Chen, X. Symbolic Discovery of Optimization Algorithms. arXiv:2302.06675

## Architecture

The optimizer implementation follows the Python MLX design:

1. **Pure TypeScript/JavaScript layer**: Optimizers are not C++ primitives but use MLX array operations
2. **Tree-based parameter handling**: Uses `treeMap` utilities to handle nested parameter structures
3. **Lazy initialization**: State is initialized on first use or explicit `init()` call
4. **Scheduler support**: Learning rate and other parameters can be functions of step count
5. **Stateful updates**: Maintains velocity/momentum in state dictionary

## Testing

Run tests with:
```bash
cd node
npm test -- test/optimizers.test.ts
```

Current tests cover:
- Constructor parameter validation
- State initialization
- Property accessors
- Error cases for invalid configurations

## Next Steps

To complete the SGD implementation:

1. **Add missing core operations** to `node/src/core/ops.ts`:
   ```typescript
   export function subtract(a: MLXArray, b: MLXArray, options?: BinaryOpOptions): MLXArray
   export function divide(a: MLXArray, b: MLXArray, options?: BinaryOpOptions): MLXArray
   ```

2. **Add scalar-array arithmetic support**:
   - Allow operations like `multiply(number, MLXArray)`
   - Support broadcasting of scalars

3. **Implement `astype()` method** on `MLXArray`:
   ```typescript
   class MLXArray {
     astype(dtype: DType): MLXArray
   }
   ```

4. **Add proper scalar construction**:
   - Support `array(5)` to create a scalar array
   - Or add `scalar(value, dtype)` helper function

5. **Complete `applySingle()` implementation** once the above are available

6. **Add integration tests** that verify gradient application with actual arrays

7. **Implement other optimizers**:
   - ✅ Lion (completed)
   - Adam
   - AdamW  
   - RMSprop
   - Adagrad
   - etc.

## References

- Python Implementation: `python/mlx/optimizers/optimizers.py`
- MLX Documentation: [https://ml-explore.github.io/mlx/](https://ml-explore.github.io/mlx/)
