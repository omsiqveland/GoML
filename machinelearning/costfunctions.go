package machinelearning

import (
	"fmt"
	"math"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// CostBinaryLogistic Tested OK
func CostBinaryLogistic(An *mat.VecDense, Y *mat.VecDense) float64 {
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
	for i := 0; i < m; i++ {
		cost += ((-math.Log(An.At(i, 0)) * Y.At(i, 0)) + (-math.Log(1-An.At(i, 0)) * (1 - Y.At(i, 0)))) / float64(m)
	}

	fmt.Println("Cost: ")
	fmt.Println(cost)

	return cost
}

// CostBinaryLogisticWithL2Regularization Testresult: 0.18584276807458602 should be 0.183984340402
func CostBinaryLogisticWithL2Regularization(An *mat.VecDense, Y *mat.VecDense, parameters map[string]*mat.Dense, lambda float64) float64 {

	m := Y.Len()
	var WArr map[string]*mat.Dense
	var bArr map[string]*mat.Dense
	WArr = make(map[string]*mat.Dense)
	bArr = make(map[string]*mat.Dense)

	for key, val := range parameters {
		if strings.HasPrefix(key, "W") {
			WArr[key] = val

		} else if strings.HasPrefix(key, "b") {
			bArr[key] = val
		}
	}
	fmt.Printf("Length = %v\n\n", len(WArr))
	crossEntropyCost := CostBinaryLogistic(An, Y)

	costL2Regularization := 0.0

	for _, val := range WArr {
		WClone := mat.DenseCopyOf(val)
		WClone.MulElem(WClone, val)
		fc := mat.Formatted(WClone, mat.Prefix("    "), mat.Squeeze())
		fmt.Printf("W = %v\n\n", fc)
		rows, cols := WClone.Dims()
		matSum := 0.0
		for WRow := 0; WRow < rows; WRow++ {
			for WCol := 0; WCol < cols; WCol++ {
				matSum += WClone.At(WRow, WCol)
			}
		}
		costL2Regularization += matSum
	}
	costL2Regularization = costL2Regularization * 0.5 * lambda / float64(m)

	fmt.Println("Regularization cost: ")
	fmt.Println(costL2Regularization)
	return crossEntropyCost + costL2Regularization
	/*	   def compute_cost_with_regularization(A3, Y, parameters, lambd):
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
