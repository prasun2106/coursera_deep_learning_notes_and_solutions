# Improving Deep Neural Networks - Hyperparametric Tuning, Optimization and Regularization - Course 2- Week1

Date Created: Jun 01, 2020 12:13 PM
Status: Courses

## W1, L1:

Iterative process:

- No. of layers
- No of hidden units
- learning rates
- activation functions etc...
1. How quickly we can make progress depends on how efficiently we can go around this cycle of iterative process. Setting up our data in train/dev/test helps us in doing so.
2. Train - 99.5%, dev - 0.25%, test - 0.25 % is also possible when we have huge number of training examples.
3. Sometimes we can also not have a test set, if we don't want an unbiased estimate of our model. Train on training set, test it multiple times on dev set to decide the neural network architecture.

## W1, L2:

1. Its possible to reduce bias as well as variance without affecting others in deepn learning. That';s why we talk less about bias variance **trade offs**.
2. Train - 1%, Dev set error - 11% → high variance
3. Train - 15%, Dev set error - 16% → High bias if human error is 0%. Model is doing fine if human error is arounf 15%.
4. Train - 15%, Dev - 30% → High bias, High variance if human error → 0%
5. Train - 0.5%, Dev - 1% → Low bias, low variance
6. This human error is called optimal error.

## W1, L3:

1. Basic recipe for machine learning:
    1. Check High Bias:
        1. This can be checked by looking at the training error. If there is very low accuracy even on train set, its a sign of high bias:
            1. Make bigger and complex network
            2. Train longer
            3. Choose different NN Architecture
    2. Check High Variance:
        1. This can be checked by looking at Dev Set Errors. If train set error is very low and dev set error is high enough, then we are dealing with the case of high variance:
            1. Get more data
            2. Regularization
            3. Choose different architecture

## W1 L4,5,6 - Regularization - L2 and L1:

1. Regularization penalizes larger weights more than the smaller weights. This can be understood in two ways:
    - In the cost function, the square term makes the effect of large w significant, and while minimizing the cost function, the algorithm penalizes large w to compensate for the contribution from them.
    - While updating the weights, we get w = (1 - learning_rate*lambda/m)w +/... → This causes decay of w and thus, the larger weights will decay more.
2. Generally speaking, the number of columns in a weight matrix is decided by the number of neurons in the previous layer, and the number of columns in a weight matrix is decided by the number of neurons in the current layer.
3. Why is Non Linearity required in Neural Networks:
    1. If we have linear activation function, there is no point of having a deep neural network as the linear activation of all the layer can be combined to give output which still can be expressed as a linear combination of all other outputs, the only difference will be updated values for weights. Proof:
        - y=h(x)=bn+Wn(bn−1+Wn−1(…(b1+W1x)…))=bn+Wnbn−1+WnWn−1bn−2+⋯+WnWn−1…W1x=b′+W′x
    2. Non linearity helps us in keeping all hidden layers alive and thus it enhances the approximation power of a neural network.
    3. Importance of ReLU:
        1. **ReLU is used to increase non linearity**
        2. its not the case that ReLU is used because its more nonlinear than tanh or sigmoid.
        3. It is used because it helps in training the neural networks much faster without compromising with generalizing accuracy.
4. The goal of regularization is to reduce w.

## W1 L7 - Dropout Regularization:

1. Consider layer 3:
    1. d3 = np.random.rand(a3.shape[0], a3.shape[1])
    2. d3 = d3 < keep_prob → d3 will be a boolean with True and False. If keep_prob = 0.8, 20% of the neurons will be kept while others will be turned off in a layer.
    3. a3 = np.multiply(a3,d3)
    4. a3 = a3/keep_prob # dropout done!
    5. New a3 will have less number of which are randomly selected
2. For each training examples, the neurons are randomly selected and also for each pass of the gradient descent, the neurons are randomly selected. So, in the first pass of grad. descent, first training example can train on neuron 1, 3,7 in layer 3, while the same training example can be trained on neuron 2,3,6,8 in the second pass of grad. descent.
3. In the fourth step, we are scaling a3 up by dividing it by keep_prob. Significance of this step - as we turn off 20 of neurons, we expect output to also get affected. So we are increasing it by multiplying it with a suitable factor.
4. Dropouts can be proved to be similar to L2 regularization, its just that the penalties are not same.

## W1 L8: Other Regularization Technique:

1. Data Augmentation - Add new examples by rotating or distorting the images
2. Early Stopping:
    1. In L2 regularization we try different values of lambda to reduce w. But in early stopping, we stop training the netwrok in midway to get the reduced value of w. w ~0 in beginning of training because we generally initialze w with smaller values. As train ing progresses, the value of w increases. For some time, the dev set error decreases with training error, but after a point of time, dev set error starts increasing as we start overfitting.
    2. We want to stop our training at that point.
    3. Orthogonalization is a concept that two operations can be done in step wise manner. Optimization can be done first which might overfit the data because of not so suitable hyperparameters. Then we take up the process of reducing overfitting in a different stage.
    4. In L2 regularization, we complete the process of optimization first. Then we reduce the overfitting.
    5. In early stopping, we mix up both the steps together. Without fully optimizing we are stopping it in mid way. → Disadvantage

![Improving%20Deep%20Neural%20Networks%20Hyperparametric%20Tun%2081bddee8a2f445e1b7e405cce30fb4bc/Untitled.png](Improving%20Deep%20Neural%20Networks%20Hyperparametric%20Tun%2081bddee8a2f445e1b7e405cce30fb4bc/Untitled.png)

## W1 L9 - Normalization:

1. x → (x-mu)
2. Its very important to normalize train and test set together, or similarly. In short, use same mu and sigma to normalize test set.
3. Why Normalization:
    1. Short Answer - Its difficult for optimization algorithm to reduce the cost function if all features are of varying scale.
    2. Explanation - Consider a simple cost function:

    $$J = \Sigma (ycapi - yi)^2 = \Sigma (ycapi - (w1x1 + bx2))^2$$

    3. If x1 is very large and x2 is very small, no matter how large the value of b, the value of cost function will stay small if w1 is kept constant. Similarly for w1 also. We will get cost function like this:

    ![Improving%20Deep%20Neural%20Networks%20Hyperparametric%20Tun%2081bddee8a2f445e1b7e405cce30fb4bc/Untitled%201.png](Improving%20Deep%20Neural%20Networks%20Hyperparametric%20Tun%2081bddee8a2f445e1b7e405cce30fb4bc/Untitled%201.png)

    4. To reach minima, learning rate should be very small, in case of unnormalized inputs.

## W1 L10 - Vanishing/Exploding Gradients:

1. In case of **deep** neural networks, the activation function increases or decreases exponentially with the power of L where L is number of layers. The value of activation function also causes the gradients to increase or decrease exponentially. In following picture, we are assuming a linear activation function just for the sake of illustration. We proved that if all of the weight matrices are slighly higher than 1, then also putting a power of 150 (number of layers) will make activation values very high. Look at the following picture:

    ![Improving%20Deep%20Neural%20Networks%20Hyperparametric%20Tun%2081bddee8a2f445e1b7e405cce30fb4bc/Untitled%202.png](Improving%20Deep%20Neural%20Networks%20Hyperparametric%20Tun%2081bddee8a2f445e1b7e405cce30fb4bc/Untitled%202.png)

## W1 L11 - Weight Initialization of Deep Network:

1.