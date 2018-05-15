package machinelearning

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// CostFunctionBinaryLogistic ...
func CostFunctionBinaryLogistic(An *mat.VecDense, Y *mat.VecDense) float64 {
	/*
		"""
			Implement the cost function

			Arguments:
			a3 -- post-activation, output of forward propagation
			Y -- "true" labels vector, same shape as a3

			Returns:
			cost - value of the cost function
			"""
			m = Y.shape[1]

			logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
			cost = 1./m * np.nansum(logprobs)

			return cost
	*/

	if An.Len() != Y.Len() {
		panic("An.Len() != Y.Len()")
	}

	m := Y.Len()

	cost := 0.0

	for i := 0; i < An.Len(); i++ {
		cost += ((math.Log(An.At(i, 0)) * Y.At(i, 0)) + (math.Log(1-An.At(i, 0)) * (1 - Y.At(i, 0)))) / float64(m)
	}

	return cost
}

// CostWithL2Regularization ...
func CostWithL2Regularization(An *mat.VecDense, Y *mat.VecDense, parameters map[string]float64, lambda float64) float64 {

	/*
	   def compute_cost_with_regularization(A3, Y, parameters, lambd):
	       """
	       Implement the cost function with L2 regularization. See formula (2) above.

	       Arguments:
	       A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
	       Y -- "true" labels vector, of shape (output size, number of examples)
	       parameters -- python dictionary containing parameters of the model

	       Returns:
	       cost - value of the regularized loss function (formula (2))
	       """
	       m = Y.shape[1]
	       W1 = parameters["W1"]
	       W2 = parameters["W2"]
	       W3 = parameters["W3"]

	       cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost

	       ### START CODE HERE ### (approx. 1 line)
	       L2_regularization_cost = (np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))*0.5*lambd/m
	       ### END CODER HERE ###

	       cost = cross_entropy_cost + L2_regularization_cost

	   	return cost
	*/
}
