# -*- coding: utf-8 -*-
"""
This module contains a TensorFlow/Keras implementation of HFCSAM
"""

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from keras.optimizers import Optimizer
from keras import backend as K
from legacy import interfaces

#############################################################################

class HFCSAM(Optimizer):
    """HFCSAM optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
    dP: float >= 0. Learning step.
    xi: float 0 < xi < 1. Angle.
    decay: float >= 0. Learning rate decay over each update.
    """

    def __init__(self, dP=0.1, xi=0.99, decay=0.0, **kwargs):
        super(HFCSAM, self).__init__(**kwargs)
        self.iterations = K.variable(0., name='iterations')
        self.first_iteration = K.variable(0., name='iterations')
        self.dP = K.variable(dP, name='dP')
        self.xi = K.variable(xi, name='xi')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def hessian_vector_product(self, ys, xs, v):
        """Multiply the Hessian of `ys` wrt `xs` by `v`.
        This is an efficient construction that uses a backprop-like approach
        to compute the product between the Hessian and another vector. The
        Hessian is usually too large to be explicitly computed or even
        represented, but this method allows us to at least multiply by it
        for the same big-O cost as backprop.
        Implicit Hessian-vector products are the main practical, scalable way
        of using second derivatives with neural networks. They allow us to
        do things like construct Krylov subspaces and approximate conjugate
        gradient descent.
        Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
        x, v)` will return an expression that evaluates to the same values
        as (A + A.T) `v`.
        Args:
        ys: A scalar value, or a tensor or list of tensors to be summed to
            yield a scalar.
        xs: A list of tensors that we should construct the Hessian over.
        v: A list of tensors, with the same shapes as xs, that we want to
           multiply by the Hessian.
        Returns:
        A list of tensors (or if the list would be length 1, a single tensor)
        containing the product between the Hessian and `v`.
        Raises:
        ValueError: `xs` and `v` have different length.
        """

        # Validate the input
        length = len(xs)
        if len(v) != length:
            raise ValueError("xs and v must have the same length.")

        # First backprop
        grads = tf.gradients(ys, xs)

        assert len(grads) == length
        elemwise_products = [
          math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
          for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
        ]

        # Second backprop
        return tf.gradients(elemwise_products, xs)



    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):

        grads = K.gradients(loss, params)
        flattenedgrads = [K.flatten(x) for x in grads]
        G = K.concatenate(flattenedgrads)

        self.updates = []

        dP = self.dP
        xi = self.xi

        if self.initial_decay > 0:
            dP *= (1. / (1. + self.decay * self.iterations))
            self.updates.append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + moments

        hessian_vector = self.hessian_vector_product(loss, params, moments)

        flattenedhvps = [K.flatten(x) for x in hessian_vector]
        F = K.concatenate(flattenedhvps)

        IGG=K.sum(G * G)
        IFF=K.sum(F * F)
        IGF=K.sum(G * F)

        dQ=-xi*dP*tf.sqrt(IGG)

        lamda2 = 0.5*tf.sqrt((IFF*IGG-IGF*IGF)/(IGG*dP*dP-dQ*dQ))

	ccond=K.greater(IGF,0.0)

        lamda1=K.switch(ccond,(-2*lamda2*dQ+IGF)/IGG, (2*lamda2*dQ+IGF)/IGG)

        for p, g, hvp, m in zip(params, grads, hessian_vector, moments):

            cond=K.greater(IFF,0.0)

            v = K.switch(cond, K.switch(ccond, -((lamda1/(2*lamda2))*g)+((1/(2*lamda2))*hvp), ((lamda1/(2*lamda2))*g)-((1/(2*lamda2))*hvp)) , -dP * g)

            self.updates.append(K.update_add(self.first_iteration, 1))

            self.updates.append(K.update(m, v))

            self.updates.append(K.update(self.xi,xi))

            new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {'dP': float(K.get_value(self.dP)),
                  'xi': float(K.get_value(self.xi)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(HFCSAM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

##############################################################################
