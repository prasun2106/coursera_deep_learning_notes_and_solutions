- If number of hyperparameters are snall, then we can create a grid.
- In deep learning, we should choose hyperparameters by random sampling instead of the grid

# Gradient descent with momentum
The momentum algorithm almost always works faster than standard gradient descent.

The simple idea is to calculate the exponentially weighted averages for your gradients and then update your weights with the new values.

Pseudo code:

vdW = 0, vdb = 0
on iteration t:
    # can be mini-batch or batch gradient descent
    compute dw, db on current mini-batch                

    vdW = beta * vdW + (1 - beta) * dW
    vdb = beta * vdb + (1 - beta) * db
    W = W - learning_rate * vdW
    b = b - learning_rate * vdb
Momentum helps the cost function to go to the minimum point in a more fast and consistent way.

beta is another hyperparameter. beta = 0.9 is very common and works very well in most cases.

In practice people don't bother implementing bias correction.