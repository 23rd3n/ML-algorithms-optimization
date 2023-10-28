## The python codes that I've written during the course period of Machine Learning and Optimization course
### 1-) [Linear Regression and Least Squares](./least\squares)
***
**1.a Linear Fit to one-dimensional data** 
* Cost function is selected as MSE (Mean Squared Error) as it can be easily
shown that MSE loss, is the optimal loss selection if noise is gaussian and
and the observation and data are jointly distributed (Maximum Likelihood Principle)
* And since the MSE loss is quadratic function, it is also convex, and since it is
convex function, it's first gradient is enough to find the global minimum, since 
for a convex function the local minimum is global minimum.
* By taking gradient with respect to the parameters and equating it to zero,
we find the optimal parameters (optimal linear fit)
***
**1.b Polynomial Fit to one-dimensional data**
* Again by finding the MSE los ||**Xa**-y||, by creating a X matrix n-by-d, we
can find the optimal parameters a, but since the matrix X itself a square matrix,
it is not directly invertible, so we use Least squares estimation to find the
optimal parameters
* `a_LS = np.linalg.inv(X@X.T)@X.T@y` or `a_LS = np.linalg.pinv(X)@y`


