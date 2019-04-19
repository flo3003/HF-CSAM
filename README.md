# A Hessian Free Neural Networks Training Algorithm with Curvature Scaled Adaptive Momentum

This is a Hessian Free Neural Networks Training Algorithm with Curvature Scaled Adaptive Momentum called HF-CSAM.

## Installation

Install via

    python setup.py install
    #pip install git+https://github.com/flo3003/HF-CSAM

``hfcsam`` requires a TensorFlow and Keras installation (the current code has been tested for realeases 1.6--1.8), but this is *not* currently enforced in the ``setup.py`` to allow for either the CPU or the GPU version.

## Usage

The ``hfcsam`` module contains the class ``HF-CSAM``, which inherits from keras optimizers and can be used as direct drop-in replacement for Keras's built-in optimizers.

    from hfcsam import HFCSAM
    
    loss = ...
    opt = HFCSAM(dP=0.07, xi=0.99)
    step = opt.minimize(loss)
    with tf.Session() as sess:
        sess.run([loss, step])

HF-CSAM has two hyper-parameters: dP and xi. The dP parameter the step size and can vary depending on the problem. In MNIST and CIFAR datasets dP is 0.05 < dP < 0.5 and  xi should be 0.5 < xi < 0.99 (the default value ``xi=0.99`` should work for most problems).

## Short Description of HF-CSAM

We give a short description of the algorithm, ignoring various details. Please refer to the [paper][1] for a complete description.

The algorithm's weight update rule is similar to SGD with momentum but with two main differences arising from the formulation of the training task as a constrained optimization problem: (i) the momentum term is scaled with curvature information (in the form of the Hessian); (ii) the coefficients for the learning rate and the scaled momentum term are adaptively determined.

The objective is to reach a minimum of the cost function L_t with respect to the synaptic weights, and simultaneously to maximize incrementally at each epoch the following quantity:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\Phi_t={\boldmath{dw}_t}^T\boldmath{H}_t\boldmath{dw}_{t-1}" title="\Large \Phi_t={\boldmath{dw}_t}^T\boldmath{H}_t\boldmath{dw}_{t-1}" />

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\boldmath{dw}_t" title="\Large \boldmath{dw}_t" /> are the weight updates at the current time step, <img src="https://latex.codecogs.com/svg.latex?\Large&space;\boldmath{dw}_{t-1}" title="\Large \boldmath{dw}_{t-1}" /> are the weight updates at the previous time step and <img src="https://latex.codecogs.com/svg.latex?\Large&space;\boldmath{H}_t" title="\Large \boldmath{H}_t" /> is the Hessian of the cost function <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}_t" title="\Large \mathcal{L}_t" />.

At each epoch t of the learning process, the vector <img src="https://latex.codecogs.com/svg.latex?\Large&space;\boldmath{w}_t" title="\Large \boldmath{w}_t" />  will be incremented by <img src="https://latex.codecogs.com/svg.latex?\Large&space;\boldmath{dw}_t" title="\Large \boldmath{dw}_t" /> , so that:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\boldmath{dw}_t^T\boldmath{dw}_t=(\delta{P})^2" title="\Large \boldmath{dw}_t^T\boldmath{dw}_t=(\delta P)^2" /> 

And the objective function <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}_t" title="\Large \mathcal{L}_t" /> must be decremented by a quantity <img src="https://latex.codecogs.com/svg.latex?\Large&space;\delta\mathcal{Q}_t" title="\Large \delta\mathcal{Q}_t" />, so that:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;d\mathcal{L}_t=\delta\mathcal{Q}_t" title="\Large d\mathcal{L}_t=\delta\mathcal{Q}_t" /> 

The learning rule can be derived by solving the following constrained optimization problem:

*Maximize     z* <img src="https://latex.codecogs.com/svg.latex?\Large&space;\Phi_t={\boldmath{dw}_t}^T\boldmath{H}_t\boldmath{dw}_{t-1}" title="\Large \Phi_t={\boldmath{dw}_t}^T\boldmath{H}_t\boldmath{dw}_{t-1}" />

*subject to the constraints*

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\boldmath{dw}_t^T\boldmath{dw}_t=(\delta{P})^2" title="\Large \boldmath{dw}_t^T\boldmath{dw}_t=(\delta P)^2" />  and <img src="https://latex.codecogs.com/svg.latex?\Large&space;d\mathcal{L}_t=\delta\mathcal{Q}_t" title="\Large d\mathcal{L}_t=\delta\mathcal{Q}_t" /> 

Hence, by solving this constrained optimization problem analytically, we get the following update rule:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\boldmath{dw}_t=-\frac{\lambda_1}{2\lambda_2}\boldmath{G}_t+\frac{1}{2\lambda_2}\boldmath{H}_td\boldmath{w}_{t-1}" title="\Large \boldmath{dw}_t=-\frac{\lambda_1}{2\lambda_2}\boldmath{G}_t+\frac{1}{2\lambda_2}\boldmath{H}_td\boldmath{w}_{t-1}" /> 

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\boldmath{G}_t" title="\Large \boldmath{G}_t" />  is the gradient of the network's loss/cost function <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}_t" title="\Large \mathcal{L}_t" />.

## Feedback

If you have any questions or suggestions regarding this implementation, please open an issue in [flo3003/HF-CSAM](https://github.com/flo3003/HF-CSAM). Apart from that, we welcome any feedback regarding the performance of HF-CSAM on your training problems (mail to flwra.sakketoy@gmail.com).

## Citation

If you use HF-CSAM for your research, please cite the [paper][1].

[1]: A Hessian Free Neural Networks Training Algorithm with Curvature Scaled Adaptive Momentum (under review)
