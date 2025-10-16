/**
 * MLX Optimizers for Node.js
 * 
 * This module provides optimizer implementations for training neural networks.
 * Based on mlx.optimizers from the Python MLX library.
 */

import MLXArray, { zeros, zeros_like } from '../core/array';
import { treeMap } from '../utils';
import { add, multiply, subtract, sign } from '../core/ops';

/**
 * Scheduler function type - maps step number to a parameter value
 */
export type Scheduler = (step: MLXArray) => MLXArray;

/**
 * Parameter value that can be either a constant or scheduled
 */
export type SchedulableParam = number | Scheduler;

/**
 * Base class for all optimizers.
 * Allows implementing an optimizer on a per-parameter basis and applying it to a parameter tree.
 */
export abstract class Optimizer {
  protected _initialized: boolean = false;
  protected _state: Record<string, any> = { step: zeros([], 'uint64') };
  protected _schedulers: Record<string, Scheduler> = {};

  constructor(schedulers?: Record<string, Scheduler>) {
    if (schedulers) {
      this._schedulers = { ...schedulers };
    }
  }

  /**
   * Initialize the optimizer's state.
   * This can be used to initialize optimizers which have state (like momentum in SGD).
   * 
   * @param parameters - A tree of parameters
   */
  init(parameters: Record<string, any>): void {
    // Initialize the optimizer state to match the parameter state
    const updateState = (params: any, state: any): any => {
      if (Array.isArray(params) || params instanceof Array) {
        const stateArray = Array.isArray(state) ? state : [];
        for (let i = 0; i < params.length; i++) {
          if (i < stateArray.length) {
            stateArray[i] = updateState(params[i], stateArray[i]);
          } else {
            stateArray.push(treeMap(() => ({}), params[i]));
          }
        }
        return stateArray;
      } else if (params !== null && typeof params === 'object' && !(params instanceof MLXArray)) {
        const stateObj = state || {};
        for (const [k, v] of Object.entries(params)) {
          if (!(k in stateObj)) {
            stateObj[k] = treeMap(() => ({}), v);
          } else {
            stateObj[k] = updateState(v, stateObj[k]);
          }
        }
        return stateObj;
      } else {
        return state;
      }
    };

    updateState(parameters, this._state);
    treeMap(
      (p: any, s: any) => {
        if (p instanceof MLXArray && (!s || Object.keys(s).length === 0)) {
          return this.initSingle(p, s || {});
        }
        return s;
      },
      parameters,
      this._state
    );
    this._initialized = true;
  }

  /**
   * Initialize the optimizer state for a single parameter.
   * To be implemented by derived classes.
   * 
   * @param parameter - A single parameter that will be optimized
   * @param state - The optimizer's state dictionary for this parameter
   */
  protected abstract initSingle(parameter: MLXArray, state: Record<string, any>): void;

  /**
   * Apply gradients to parameters and return updated parameters.
   * 
   * @param gradients - A tree of gradients
   * @param parameters - A tree of parameters
   * @returns Updated parameters
   */
  applyGradients(gradients: Record<string, any>, parameters: Record<string, any>): Record<string, any> {
    if (!this._initialized) {
      this.init(gradients);
    }

    // Update any scheduled variables
    for (const [param, scheduler] of Object.entries(this._schedulers)) {
      this._state[param] = scheduler(this.step);
    }

    // Increment the step
    const currentStep = this.step.toTypedArray()[0] as number;
    this._state.step = zeros([], 'uint64');
    // Note: We would need to set the actual value here but that requires array construction from scalar
    // For now, this is a placeholder that needs proper implementation

    // Apply the update
    return treeMap(
      (gradient: any, parameter: any, state: any) => {
        if (gradient instanceof MLXArray && parameter instanceof MLXArray) {
          return this.applySingle(gradient, parameter, state);
        }
        return parameter;
      },
      gradients,
      parameters,
      this._state
    ) as Record<string, any>;
  }

  /**
   * Apply the optimizer update to a single parameter.
   * To be implemented by derived classes.
   * 
   * @param gradient - The gradient for this parameter
   * @param parameter - The parameter to update
   * @param state - The optimizer's state for this parameter
   * @returns Updated parameter
   */
  protected abstract applySingle(
    gradient: MLXArray,
    parameter: MLXArray,
    state: Record<string, any>
  ): MLXArray;

  /**
   * Get the optimizer's state dictionary
   */
  get state(): Record<string, any> {
    return this._state;
  }

  /**
   * Set the optimizer's state dictionary
   */
  set state(state: Record<string, any>) {
    this._initialized = false;
    this._state = state;
  }

  /**
   * Get the current step count
   */
  get step(): MLXArray {
    return this._state.step;
  }

  /**
   * Get the learning rate
   */
  get learningRate(): MLXArray {
    return this._state.learning_rate;
  }

  /**
   * Set the learning rate
   */
  set learningRate(lr: number | MLXArray) {
    if (typeof lr === 'number') {
      // Create a scalar array - this needs proper scalar construction support
      this._state.learning_rate = zeros([]);
    } else {
      this._state.learning_rate = lr;
    }
  }

  /**
   * Helper to optionally put a parameter on a schedule
   */
  protected _maybeSchedule(name: string, param: SchedulableParam): void {
    if (typeof param === 'function') {
      this._schedulers[name] = param;
      const parameter = param(this.step);
      this._state[name] = parameter;
    } else {
      // Create scalar array from number - needs proper scalar construction
      this._state[name] = zeros([]);
    }
  }
}

/**
 * SGD Optimizer Options
 */
export interface SGDOptions {
  /** The learning rate */
  learningRate: SchedulableParam;
  /** The momentum strength (default: 0) */
  momentum?: number;
  /** The weight decay (L2 penalty) (default: 0) */
  weightDecay?: number;
  /** Dampening for momentum (default: 0) */
  dampening?: number;
  /** Enables Nesterov momentum (default: false) */
  nesterov?: boolean;
}

/**
 * The stochastic gradient descent optimizer.
 * 
 * Updates a parameter w with a gradient g as follows:
 * 
 *   v_{t+1} = μ * v_t + (1 - τ) * g_t
 *   w_{t+1} = w_t - λ * v_{t+1}
 * 
 * where λ is the learning rate, μ is the momentum strength, and τ is the dampening.
 * 
 * @example
 * ```typescript
 * const optimizer = new SGD({ learningRate: 0.01, momentum: 0.9 });
 * // ... during training:
 * // const updatedParams = optimizer.applyGradients(gradients, parameters);
 * ```
 */
export class SGD extends Optimizer {
  momentum: number;
  weightDecay: number;
  dampening: number;
  nesterov: boolean;

  constructor(options: SGDOptions) {
    super();

    const { learningRate, momentum = 0.0, weightDecay = 0.0, dampening = 0.0, nesterov = false } = options;

    if (nesterov && (momentum <= 0 || dampening !== 0)) {
      throw new Error('Nesterov momentum requires a momentum and zero dampening.');
    }

    this._maybeSchedule('learning_rate', learningRate);
    this.momentum = momentum;
    this.weightDecay = weightDecay;
    this.dampening = dampening;
    this.nesterov = nesterov;
  }

  protected initSingle(parameter: MLXArray, state: Record<string, any>): void {
    // Initialize velocity with zeros_like
    state.v = zeros_like(parameter);
  }

  protected applySingle(
    gradient: MLXArray,
    parameter: MLXArray,
    state: Record<string, any>
  ): MLXArray {
    // Note: This is a simplified implementation that demonstrates the structure.
    // The full implementation requires additional operations that are not yet
    // available in the Node.js bindings:
    // - Subtraction operator
    // - Scalar multiplication with arrays
    // - astype() for dtype conversion
    //
    // This code will need to be updated once those operations are available.
    
    throw new Error(
      'SGD.applySingle is not yet fully implemented. ' +
      'This requires additional core operations (subtract, scalar ops, etc.) ' +
      'that are not yet available in the Node.js MLX bindings.'
    );

    // This is what the implementation should look like once operations are available:
    /*
    let grad = gradient;
    
    // Apply weight decay if configured
    if (this.weightDecay !== 0) {
      grad = add(grad, multiply(array(this.weightDecay), parameter));
    }

    // If no momentum, do simple update
    if (this.momentum <= 0) {
      const lr = this.learningRate; // Would need .astype(gradient.dtype) 
      return subtract(parameter, multiply(lr, grad));
    }

    // Get velocity from state
    let v = state.v || zeros_like(parameter);
    
    // Update velocity
    v = multiply(array(this.momentum), v);
    if (this.dampening > 0) {
      v = add(v, multiply(array(1 - this.dampening), grad));
    } else {
      v = add(v, grad);
    }

    // Compute update
    let update: MLXArray;
    if (this.nesterov) {
      update = add(grad, multiply(array(this.momentum), v));
    } else {
      update = v;
    }

    // Store velocity
    state.v = v;

    // Apply update
    const lr = this.learningRate; // Would need .astype(gradient.dtype)
    return subtract(parameter, multiply(lr, update));
    */
  }
}

/**
 * Lion Optimizer Options
 */
export interface LionOptions {
  /** The learning rate */
  learningRate: SchedulableParam;
  /** The coefficients (beta1, beta2) used for computing gradient momentum and update direction (default: [0.9, 0.99]) */
  betas?: [number, number];
  /** The weight decay (default: 0.0) */
  weightDecay?: number;
}

/**
 * The Lion optimizer.
 * 
 * Lion (EvoLved Sign Momentum) is an optimizer that uses the sign operation for updates,
 * which tends to produce updates with larger norm than other optimizers like SGD and Adam.
 * 
 * We recommend a learning rate that is 3-10x smaller than AdamW and a weight decay 
 * 3-10x larger than AdamW to maintain the strength (lr * wd).
 * 
 * Reference: Chen, X. Symbolic Discovery of Optimization Algorithms. 
 * arXiv preprint arXiv:2302.06675.
 * 
 * Update formula:
 *   c_{t+1} = β₁ * m_t + (1 - β₁) * g_t
 *   m_{t+1} = β₂ * m_t + (1 - β₂) * g_t
 *   w_{t+1} = w_t - η * (sign(c_t) + λ * w_t)
 * 
 * where η is the learning rate, β₁ and β₂ are the beta coefficients,
 * and λ is the weight decay.
 * 
 * @example
 * ```typescript
 * const optimizer = new Lion({ learningRate: 0.0001, betas: [0.9, 0.99], weightDecay: 0.01 });
 * // ... during training:
 * // const updatedParams = optimizer.applyGradients(gradients, parameters);
 * ```
 */
export class Lion extends Optimizer {
  betas: [number, number];
  weightDecay: number;

  constructor(options: LionOptions) {
    super();

    const { learningRate, betas = [0.9, 0.99], weightDecay = 0.0 } = options;

    this._maybeSchedule('learning_rate', learningRate);
    this.betas = betas;
    this.weightDecay = weightDecay;
  }

  protected initSingle(parameter: MLXArray, state: Record<string, any>): void {
    // Initialize momentum with zeros_like
    state.m = zeros_like(parameter);
  }

  protected applySingle(
    gradient: MLXArray,
    parameter: MLXArray,
    state: Record<string, any>
  ): MLXArray {
    // Get learning rate (Note: In Python, lr is cast to gradient.dtype using astype)
    // For now, we use the learning rate as-is since it's stored as an MLXArray
    const lr = this.learningRate;
    const [b1, b2] = this.betas;
    const weightDecay = this.weightDecay;

    // Get momentum from state
    const m = state.m;

    // Compute c = b1 * m + (1 - b1) * gradient
    const c = add(multiply(b1, m), multiply(1 - b1, gradient));

    // Update momentum: m = b2 * m + (1 - b2) * gradient
    state.m = add(multiply(b2, m), multiply(1 - b2, gradient));

    // Apply weight decay if configured
    let updatedParameter = parameter;
    if (weightDecay > 0) {
      // parameter = (1 - lr * weight_decay) * parameter
      updatedParameter = multiply(subtract(1, multiply(lr, weightDecay)), parameter);
    }

    // Compute final update: parameter - lr * sign(c)
    return subtract(updatedParameter, multiply(lr, sign(c)));
  }
}

export default {
  Optimizer,
  SGD,
  Lion,
};
