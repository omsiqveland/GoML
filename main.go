package main

import (
	"GoML/machinelearning"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	//test := mat.NewVecDense(3, []float64{1, 2, 3})

	//calc := BiasCorrection(test, 5.0)

	A3 := mat.NewVecDense(5, []float64{0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402})
	Y := mat.NewVecDense(5, []float64{1, 1, 0, 1, 0})

	var parameters map[string](*mat.Dense)
	parameters = make(map[string](*mat.Dense))
	parameters["W1"] = mat.NewDense(2, 3, []float64{1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763, -2.3015387})
	parameters["W2"] = mat.NewDense(3, 2, []float64{0.3190391, -0.24937038, 1.46210794, -2.06014071, -0.3224172, -0.38405435})
	parameters["W3"] = mat.NewDense(1, 3, []float64{-0.87785842, 0.04221375, 0.58281521})

	fmt.Println(machinelearning.CostBinaryLogisticWithL2Regularization(A3, Y, parameters, 0.1))
	fmt.Println(machinelearning.Relu(3))
}
