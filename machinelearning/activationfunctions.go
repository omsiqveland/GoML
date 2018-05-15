package machinelearning

import (
	"math"
)

func Sigmoid(z float64) float64 {
	// 1 / 1 + e^(-z)
	result := 1 / (1 + math.Exp(-z))
	return result
}

/* Relu ...
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
	"""
*/
// Relu computes the relu of x
func Relu(z float64) float64 {
	// 1 / 1 + e^(-z)
	result := math.Max(z, 0)
	return result
}

func softmax() {

}
