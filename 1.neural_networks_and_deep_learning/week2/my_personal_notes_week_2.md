# Neural Network and Deep Learning - Course 1

Date Created: May 28, 2020 9:58 PM
Status: Courses

## Gradient Descent for Logistic Regression:

Simple steps for ith training example [all i are in superscript]:

1. [w1,x1,w2,x2,b] → zi = w1*x1i + w2*x2i + b → ai = sigmoid (zi) → L (ai, yi) = -yi(ln(ai)) - (1-yi)(ln(1-ai))
2. Differential will be calculated from right to left.
3. dL/da = -(yi/ai) + ((1-yi)/(1-ai))
4. dL/dz = dL/da*da/dz = ai - yi [after simplification]
5. dL/dw1 [our aim is to find dw so that we can apply gradient descent] = [dL/da*da/dz]*dz/dw1 = dL/dz*dz/dw1 = (ai - yi) * x1 [as dz/dw1 = x1]
6.