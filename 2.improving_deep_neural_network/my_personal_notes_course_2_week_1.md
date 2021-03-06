# Improving Deep Neural Networks - Hyperparametric Tuning, Optimization and Regularization - Course 2- Week1

Date Created: Jun 01, 2020 12:13 PM
Status: Courses

# Week 1 Lectures:

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

- A partial solution to the Vanishing / Exploding gradients in NN is better or more careful choice of the random initialization of weights
- In a single neuron (Perceptron model): `Z = w1x1 + w2x2 + ... + wnxn`
    - So if `n_x` is large we want `W`'s to be smaller to not explode the cost.
- So it turns out that we need the variance which equals `1/n_x` to be the range of `W`'s
- Please note that weights in all the layers need to be initialized randomly.
- So lets say when we initialize `W`'s like this (better to use with `tanh` activation):or variation of this (Bengio et al.): here n[l-1] represents number of inputs to the layer.

    ```python
    np.random.rand(shape) * np.sqrt(1/n[l-1])
    ```

    ```python
    np.random.rand(shape) * np.sqrt(2/(n[l-1] + n[l]))
    ```

- Setting initialization part inside sqrt to `2/n[l-1]` for `ReLU` is better:

    ```python
    np.random.rand(shape) * np.sqrt(2/n[l-1])
    ```

- Number 1 or 2 in the numerator can also be a hyperparameter to tune (but not the first to start with)
- This is one of the best way of partially solution to Vanishing / Exploding gradients (ReLU + Weight Initialization with variance) which will help gradients not to vanish/explode too quickly
- The initialization in this video is called "He Initialization / Xavier Initialization" and has been published in 2015 paper.
- Refer this:

[Weight Initialization in Neural Networks: A Journey From the Basics to Kaiming](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)

# Week 1 Assignments:

## 1. Initialization:

### Zero Initialization:

- In general, initializing all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with n[l]=1n[l]=1 for every layer, and the network is no more powerful than a linear classifier such as logistic regression.
- The weights W[l]W[l] should be initialized randomly to break symmetry. It is however okay to initialize the biases b[l]b[l] to zeros. Symmetry is still broken so long as W[l]W[l] is initialized randomly.
- To break symmetry, lets intialize the weights randomly. Following random initialization, each neuron can then proceed to learn a different function of its inputs. In this exercise, you will see what happens if the weights are intialized randomly, but to very large values.

### Random initialization with large values:

- The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when , the loss goes to infinity.

    log(a[3])=log(0)

    log⁡(a[3])=log⁡(0)

- `w = np.random.randn(layers_dims [L], layers_dims [L-1]) * 10` - Code for initializing weights of Lth layer. 10 is the factor with which we multiplied our weights to scale it to a larger values.
- Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
- If you train this network longer you will see better results, but initializing with overly large random numbers slows down the optimization.
- So, we will try to initialize with a smaller value. How small? Researchers have published few papers as recent as 2015 to answer these questions.

### He Initialization - Random initial values with a specific scaling factor:

- Instead of 10, we multiply random initial values of weights with

$$\sqrt{\frac{2}{Dimension of previous layer}} $$

### **What you should remember from this initialization assignment**:

Different initializations lead to different resultsRandom initialization is used to break symmetry and make sure different hidden units can learn different thingsDon't intialize to values that are too largeHe initialization works well for networks with ReLU activations.

## 2. Regularization:

**Observations**:

- The value of lambda  is a hyperparameter that you can tune using a dev set.
- L2 regularization makes your decision boundary smoother. If  is too large, it is also possible to "oversmooth", resulting in a model with high bias.

**What is L2-regularization actually doing?**:

- L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

**What you should remember** -- the implications of L2-regularization on:

- The cost computation:A regularization term is added to the cost
- The backpropagation functionv:There are extra terms in the gradients with respect to weight matricesWeights end up smaller ("weight decay"):Weights are pushed to smaller values.

### Dropout:

When you shut some neurons down, you actually modify your model. The idea behind drop-out is that at each iteration, you train a different model that uses only a subset of your neurons. With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time.

### Dropout **Notes**:

- A **common mistake** when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training.
- Deep learning frameworks like [tensorflow](https://www.tensorflow.org/api_docs/python/tf/nn/dropout), [PaddlePaddle](http://doc.paddlepaddle.org/release_doc/0.9.0/doc/ui/api/trainer_config_helpers/attrs.html), [keras](https://keras.io/layers/core/#dropout) or [caffe](http://caffe.berkeleyvision.org/tutorial/layers/dropout.html) come with a dropout layer implementation. Don't stress - you will soon learn some of these frameworks.

**What you should remember about dropout:**

Dropout is a regularization technique.You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.Apply dropout both during forward and backward propagation.During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.

## 3. Gradient Checking