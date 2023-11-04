## The Python codes that I've written during the course period of Machine Learning and Optimization course
### 1-) [Linear Regression and Least Squares](./1-LinearRegression)
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

### 2-) [Fitting a wrong model and analyzing the prediction error](./2-PredictionError)
***
* In linear regression, when we fit polynomials, polynomial functions can represent linear models, therefore we don't have a bias error. Thus as we solved in question 4, the prediction error is only the part coming from the noise vector. The prediction error scales linearly with D and this can be seen on the plot of part 5, as D increases, the prediction error also increases for the same number of observations.

* However, in problem 6, polynomial fit cannot represent the exponential function; thus, we have an irreducible error (bias term), therefore prediction error decreases with increasing D value, in contrast to problem 5. However, when we increase D, overfitting to noise starts therefore, there is a trade-off between bias and variance terms.

* The variance (overfitting) will decrease with increasing n, therefore at some point bias will be the only error term.

