## W2 L1 - Mini-Batch Gradient Descent

- Epoch - Single pass through the training set.
    * In batch gradient descent, 1 epoch allows us to take one gradient descent step
    * In mini - batch gradient descent, 1 epoch allows us to take 5000 gradient descent step (or, the steps will be equal to the number of mini batches)
- Essentially everything remains same, except our set of training example has been divided into many batches. And we will update w and b after each iteration of mini batch step.

## W2 L2 - Understanding mini-batch gradient descent

- Cost function of batch gradient descent keeps on descreasing with each iteration (it increases only if learning rate is too big)
- In the case of mini batch gradient descent, it keeps on increasing or decreasing, but overall it decreases.
- Two extremes:
    * mini batch size = m -> batch gradient descent -> Disadvantage - too large dataset to process at each iteration
    * mini batch size = 1 -> stochastic gradient descent -> Disadvantage - extremely noisy, will never converge, cant take benefit of speeding up provided by vectorization

- Guidelines for selecting batch size:
    * small training examples (m<=2000) -> use batch gradient descent
    * for large training set -> m = 64, 128, 256, 512, <- more common 1024...
    * in practice, batch size is also a hyperparameter
    
## W2 L3 - Exponentially