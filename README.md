## The Python codes that I've written during the course period of Machine Learning and Optimization course
### 1-) [Linear Regression and Least Squares](./1-LinearRegression/hw1_coding.ipynb)
***
**1.a Linear Fit to one-dimensional data** 
* Cost function is selected as MSE (Mean Squared Error) as it can be easily
shown that MSE loss is the optimal loss selection if the noise is Gaussian and
and the observation and data are jointly distributed (Maximum Likelihood Principle)
* And since the MSE loss is a quadratic function, it is also convex, and since it is
convex function, its first gradient is enough to find the global minimum, since 
for a convex function, the local minimum is the global minimum.
* By taking the gradient with respect to the parameters and equating it to zero,
we find the optimal parameters (optimal linear fit)
***
**1.b Polynomial Fit to one-dimensional data**
* Again by finding the MSE loss ||**Xa**-y||, by creating a X matrix n-by-d, we
can find the optimal parameters a, but since the matrix X itself a square matrix,
it is not directly invertible, so we use Least squares estimation to find the
optimal parameters
* `a_LS = np.linalg.inv(X@X.T)@X.T@y` or `a_LS = np.linalg.pinv(X)@y`
* However, in order for the LS estimate to be unique, X must have full column rank (rows must be linearly independent)
***

### 2-) [Fitting a wrong model and analyzing the prediction error](./2-PredictionError/hw2_coding.ipynb)
***
* In linear regression, when we fit polynomials, polynomial functions can represent linear models, therefore we don't have a bias error. Thus as we solved in question 4, the prediction error is only the part coming from the noise vector. The prediction error scales linearly with D and this can be seen on the plot of part 5, as D increases, the prediction error also increases for the same number of observations.

* However, in problem 6, polynomial fit cannot represent the exponential function; thus, we have an irreducible error (bias term), therefore prediction error decreases with increasing D value, in contrast to problem 5. However, when we increase D, overfitting to noise starts therefore, there is a trade-off between bias and variance terms.

* The variance (overfitting) will decrease with increasing n, therefore at some point bias will be the only error term.
***
### 3-) [Ridge Regression analysis and 5-fold cross validation for baseball](./3-RidgeRegCrossVal/hw3_me.ipynb)

* Importance of normalization is investigated, having large scales reduces the scales of the weigths, that doesn't change the solution. However, when the regularization is also included then there will be a bias towards larger weigths. Therefore normalize the features before working with them so that all features will be behaved equally.
* Thus, we included the bias term in feature matrix, but not on the coefficents. So that, we account for the bias term but we don't change the optimization proecudre for the bias term.
* After, we obtained a closed-form solution for Ridge-regression, given that feature matrix has full column rank, n>>d, it is unique, we plotted the effect of lamda on the  norm of coefficients. As the lamda increases, due to min-opitmization, norm of the coefficients decreases because they are punished more.
* We verified, for lamda = 0, ridge=least squares; and for lamda=inf, ridge = 0. Because having infinite lamda means that any norm on the coefficients will make the optimization impossible therefore for a feasible solution coefficeints must be zero.
* Then, after closed form solution, we used 5-fold cross validation to find the best lamda (a similar approach to fine-tuning)
* After we find the best lamda value, we used that lamda to calculate the best ridge estimate and linked them with the actual baseball data to interpret them what those coefficients mean.

### 4-)[Gradient Descent Analysis for diffferent step sizes and loss functions](./4-GradientDescentAnalysis/hw4.ipynb)
***
* For norm function, constant step-size cannot converge to minima but instead oscillate at some level. However, for squared-norm, which is the most popular loss function can converge to 1% after 20 many iterations with a constant step size of 1.

* If we use a exponentially decreasing step function, then both norms cannot reach to minima since the step sizes will be too small after some iterations.

* However, if we use linearly decreasing step function then both norms reach the solution but they converge at different rates, this rate is also dependent upon the gradient at the actual step. The graphs can be analised inside the python notebook.

### 6-) [Anaylsis of Stochastic and Batch Gradient Method](./6-StochasticGradientMethod/hw6_v2.ipynb)
***
* For some functions, which are convex but non-differantiable, the gradient methods cannot be exploited. However, this is not stopping us from using the sub-gradients which are equal to the gradient of the function if the function are differantiable. In this notebook, we use subgradients for hinge loss L(theta,x) = sum(max(0, 1,-xi* yi * theta)). Hinge loss is not differantiable, however, it has subgradients.

* Hinge loss corresponds to soft-margin support vector machines.

* If we have huge train set size, then computing the gradient of the loss function at every step is somewhere around O(n), which can be quite huge since we have to wait to make a step at each epoch. However if we use stochastic gradient method, which uses the unbiased estimator of the gradient function itself. We make a step at each data point, and it is widely known that, these steps will move us closer to minimum on average. That means, we can go away from the minima at some steps. However, instead of having O(n) at each step we have approximately O(1).

* In the case of subgradients, SGM converges faster than GD.

### 7-) [Anaylsis of Algorithmic Regularization(Early-stopping)](./8-EarlyStopping/logistic_regression_with_early_stopping.ipynb)
***
* As it can be seen from the plots, in the less-parametrized regime, we observe overfitting much more, it reduces the test accuracy therefore we should apply algorithmic regularization (early-stopping) so that we don't overfit. In the over-parametrized regime, we get higher test accuracy and less overfitting for the same number of epochs. This is because when the number of parameters are much higher than the size of the dataset, our model is less-sensitive the changes in the data.

* If we have more parameters than the size of the data-set, the model performs better.

### 8-) [Fully Connected Multi Layer Perceptron](./10-FullyConnectedNN/neural_networks_pytorch.ipynb)
***
* I've used CIFAR-10 dataset with 3000 training and 1000 test sample to classify cat, dog and ship. Here images have a dimension of 4x3x32x32, indicating we used batch_size of 4 and rgb images with 32x32 resolution. In our simple neural network architecture we have only one hidden layer with flattened input vector and output layer who outputs the logits of the classes. Then we use cross-entropy loss to calculate the values of the classes. Then the maximum value is the predicted class.

* We observe that ship has the best classification accuracy since it is more easy to distinguish from cats and dogs since it has very different structure than the cats and dogs who have very similar features (tail, ears, eyes etc.)

### 9-) [Convolutional Neural Networks](./11-ConvolutionalNeuralNetworks/cnn.ipynb)
In the previous homework we obtained the following results:
-   Overall Test Accuracy: 0.6790
-   Test Accuracy for cat: 0.6220
-   Test Accuracy for dog: 0.5123
-   Test Accuracy for ship: 0.8879
-   The classifier performs best on the class "ship"

And here, with CNN, we obtain an overall test accuracy of 0.6416, which is very comparable to FCNN. This might suggest an upperbound on the performance since we are getting the same performance with two different architectures.

* As we can see, when we apply shuffling operation, it reduces the accuracy drastically. The explanation for that phennomena is because shuffling the pixels disrupts spatial information in the images, making it challenging for models to learn meaningful patterns. For CNNs, which inherently leverage spatial hierarchies, shuffling can significantly degrade performance.

* When we shuffle the pixels on the previous homework's code, we see that it is still in the order of 0.65 does not affect the accuracy since all of the pixels are connected and FCNN does not make any use of spatial information, it is basically the same thing for FCNN.
****
