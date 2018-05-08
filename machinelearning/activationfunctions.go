package machinelearning

import (
	"math"
)

func Sigmoid(z float64) float64 {
	// 1 / 1 + e^(-z)
	result := 1 / (1 + math.Exp(-z))
	return result
}

func Relu(z float64) float64 {
	// 1 / 1 + e^(-z)
	result := math.Max(z, 0)
	return result
}

func softmax() {

}
