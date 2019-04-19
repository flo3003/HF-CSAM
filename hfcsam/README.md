# A Hessian Free Neural Networks Training Algorithm with Curvature Scaled Adaptive Momentum

This is a Hessian Free Neural Networks Training Algorithm with Curvature Scaled Adaptive Momentum called HF-CSAM.

## Installation

Install via

    python setup.py install
    #pip install git+https://github.com/

``hfcsam`` requires a TensorFlow and Keras installation (the current code has been tested for realeases 1.6--1.8), but this is *not* currently enforced in the ``setup.py`` to allow for either the CPU or the GPU version.

## Usage

The ``hfcsam`` module contains the class ``HF-CSAM``, which inherits from keras optimizers and can be used as direct drop-in replacement for Keras's built-in optimizers.

    from hfcsam import HFCSAM
    
    loss = ...
    opt = HFCSAM(dP=0.07, xi=0.99)
    step = opt.minimize(loss)
    with tf.Session() as sess:
        sess.run([loss, step])

HF-CSAM has two hyper-parameters: dP and xi. The dP parameter the step size and can vary depending on the problem. In MNIST and CIFAR datasets dP is 0.05 < δP < 0.5 and  xi should be 0.5 < xi < 0.99 (the default value ``xi=0.99`` should work for most problems).

## Short Description of HF-CSAM

We give a short description of the algorithm, ignoring various details. Please refer to the [paper][1] for a complete description.

The algorithm's weight update rule is similar to SGD with momentum but with two main differences arising from the formulation of the training task as a constrained optimization problem: (i) the momentum term is scaled with curvature information (in the form of the Hessian); (ii) the coefficients for the learning rate and the scaled momentum term are adaptively determined.

The objective is to reach a minimum of the cost function L_t with respect to the synaptic weights, and simultaneously to maximize incrementally at each epoch the following quantity:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\Phi = dw_t^T * H * dW_{t-1}" title="\Large \Phi = dw_t^T * H * dW_{t-1}" />

\Phi = dw_t^T * H * dW_{t-1}
 
where dw_t are the weight updates at the current time step, dw_{t=1} are the weight updates at the previous time step and H is the Hessian of the cost function L_t.

At each epoch t of the learning process, the vector w_t will be incremented by dw_t, so that:

dw_t^T * dw_t = dP^2

And the objective function L_t must be decremented by a quantity dQ_t, so that:

dL_t = dQ_t

The learning rule can be derived by solving the following constrained optimization problem:

Maximize dw_t^T * H * dW_{t-1}

subject to the constraints 

dw_t^T * dw_t = dP^2 and 
       dL_t = dQ_t

Hence, by solving this constrained optimization problem analytically, we get the following update rule:

dw_t = - \lambda1 / (2 * \lambda2) * Gt + 1 / (2 * \lambda2) * H * dw_{t-1}

where G_t is the gradient of the network's loss/cost function L_t

## Feedback

If you have any questions or suggestions regarding this implementation, please open an issue in [](https://github.com/). Apart from that, we welcome any feedback regarding the performance of HF-CSAM on your training problems (mail to flwra.sakketoy@gmail.com).

## Citation

If you use HF-CSAM for your research, please cite the [paper][1].

[1]: https://arxiv.org/abs/


