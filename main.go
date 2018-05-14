package main

import (
	"GoML/machinelearning"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	test := mat.NewVecDense(3, []float64{1, 2, 3})

	calc := BiasCorrection(test, 5.0)
	fmt.Println(calc)
	fmt.Println(machinelearning.Relu(3))
}
