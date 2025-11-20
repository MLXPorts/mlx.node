.. _nodejs_devices_and_streams:

Devices and Streams
===================

MLX for Node.js provides control over execution devices and computation streams,
allowing you to manage where and how operations are executed.

Devices
-------

By default, MLX for Node.js uses the Metal GPU as the default device on Apple
Silicon. Operations automatically execute on the GPU without explicit device
management.

.. code-block:: typescript

   import { array, add } from 'mlx.node';

   // Operations run on GPU by default
   const a = array([1, 2, 3], [3], 'float32');
   const b = array([4, 5, 6], [3], 'float32');
   const c = add(a, b);  // Executes on Metal GPU

Device Function
^^^^^^^^^^^^^^^

**device(deviceType)**
  Get or set the device for operations.

  :param deviceType: 'gpu' or 'cpu'
  :returns: Device object

  .. code-block:: typescript

     import { core } from 'mlx.node';

     const gpu = core.device('gpu');
     const cpu = core.device('cpu');

Streams
-------

Streams allow you to manage and schedule operations on different execution
queues. This is useful for overlapping computation and data transfers or
managing concurrent operations.

Default Stream
^^^^^^^^^^^^^^

**defaultStream()**
  Get the default stream for the current device.

  .. code-block:: typescript

     import { defaultStream } from 'mlx.node';

     const stream = defaultStream();

Creating New Streams
^^^^^^^^^^^^^^^^^^^^

**newStream(device)**
  Create a new stream on the specified device.

  :param device: Device object or device type string
  :returns: Stream object

  .. code-block:: typescript

     import { newStream, device } from 'mlx.node';

     const gpu = device('gpu');
     const stream1 = newStream(gpu);
     const stream2 = newStream(gpu);

Setting the Default Stream
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**setDefaultStream(stream)**
  Set the default stream for operations.

  :param stream: Stream object to set as default

  .. code-block:: typescript

     import { newStream, setDefaultStream } from 'mlx.node';

     const customStream = newStream('gpu');
     setDefaultStream(customStream);

Synchronization
^^^^^^^^^^^^^^^

**synchronize(stream)**
  Wait for all operations on a stream to complete.

  :param stream: Optional stream to synchronize (default: all streams)

  .. code-block:: typescript

     import { array, add, synchronize, newStream } from 'mlx.node';

     const stream = newStream('gpu');
     const a = array([1, 2, 3], [3], 'float32');
     const b = array([4, 5, 6], [3], 'float32');
     const c = add(a, b);

     // Wait for operations to complete
     synchronize(stream);

Stream Contexts
---------------

Stream contexts allow you to temporarily use a different stream for a block of
operations.

**streamContext(stream)**
  Create a context for executing operations on a specific stream.

  :param stream: Stream object
  :returns: Context object

  .. code-block:: typescript

     import { core, newStream } from 'mlx.node';

     const stream1 = newStream('gpu');
     const stream2 = newStream('gpu');

     // Operations on stream1
     const ctx1 = core.streamContext(stream1);
     const a = core.array([1, 2, 3], [3], 'float32');
     // ... operations ...

     // Switch to stream2
     const ctx2 = core.streamContext(stream2);
     const b = core.array([4, 5, 6], [3], 'float32');
     // ... operations ...

**withStream(stream, fn)**
  Execute a function with a specific stream as the default.

  :param stream: Stream object
  :param fn: Function to execute
  :returns: Result of the function

  .. code-block:: typescript

     import { withStream, newStream, array, add } from 'mlx.node';

     const customStream = newStream('gpu');

     const result = withStream(customStream, () => {
       const a = array([1, 2, 3], [3], 'float32');
       const b = array([4, 5, 6], [3], 'float32');
       return add(a, b);
     });

The Stream Class
----------------

Stream objects represent execution queues for operations.

Stream Properties
^^^^^^^^^^^^^^^^^

**device**
  The device associated with this stream.

  .. code-block:: typescript

     import { newStream } from 'mlx.node';

     const stream = newStream('gpu');
     console.log(stream.device);  // 'gpu'

Use Cases
---------

Overlapping Operations
^^^^^^^^^^^^^^^^^^^^^^

Use multiple streams to overlap independent operations:

.. code-block:: typescript

   import { newStream, withStream, array, add, multiply } from 'mlx.node';

   const stream1 = newStream('gpu');
   const stream2 = newStream('gpu');

   // Start computation on stream1
   const result1 = withStream(stream1, () => {
     const a = array([1, 2, 3], [3], 'float32');
     const b = array([4, 5, 6], [3], 'float32');
     return add(a, b);
   });

   // Concurrently compute on stream2
   const result2 = withStream(stream2, () => {
     const c = array([7, 8, 9], [3], 'float32');
     const d = array([10, 11, 12], [3], 'float32');
     return multiply(c, d);
   });

React/Next.js Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Streams are particularly useful in server-side rendering contexts:

.. code-block:: typescript

   import { streaming, newStream, withStream } from 'mlx.node';

   // Create a dedicated stream for SSE responses
   const inferenceStream = newStream('gpu');

   async function* generateTokens() {
     await withStream(inferenceStream, async () => {
       // Model inference operations on dedicated stream
       // ...
     });
   }

   // In a Next.js route handler
   export async function GET() {
     return streaming.eventStreamResponse(generateTokens());
   }

Best Practices
--------------

1. **Use default stream for simple cases**: For most applications, the default
   stream is sufficient.

2. **Create dedicated streams for isolation**: Use separate streams when you
   need to isolate or prioritize certain computations.

3. **Synchronize when needed**: Only synchronize when you need to ensure
   operations have completed before proceeding.

4. **GPU is default**: Operations automatically run on Metal GPU unless
   explicitly specified otherwise.

5. **CPU fallback**: Use CPU device for operations that don't benefit from GPU
   acceleration or for debugging.

See Also
--------

- :ref:`nodejs_array` - Array creation and manipulation
- :ref:`nodejs_ops` - Array operations
- :ref:`Node/React Streaming Architecture <node_streaming>` - Server-side streaming integration
