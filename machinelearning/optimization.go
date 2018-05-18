package machinelearning

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// BiasCorrection ...
func BiasCorrection(value *mat.VecDense, beta float64) *mat.VecDense {
	result := mat.NewVecDense(value.Len(), nil)
	result.ScaleVec(1/(1-math.Pow(beta, 2)), value)
	return result
}

// LinearWeightedAverage ...
func LinearWeightedAverage() {
	//BiasCorrection()
}

// GradientDescentWithMomentum ...
func GradientDescentWithMomentum() {
	//BiasCorrection()
}

// GradientChecking ...
func GradientChecking() {
	// DEGUG ONLY - slow computation
	numericalApprox()

	/*
				def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
		    """
		    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

		    Arguments:
		    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
		    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
		    x -- input datapoint, of shape (input size, 1)
		    y -- true "label"
		    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

		    Returns:
		    difference -- difference (2) between the approximated gradient and the backward propagation gradient
		    """

		    # Set-up variables
		    parameters_values, _ = dictionary_to_vector(parameters)
		    grad = gradients_to_vector(gradients)
		    num_parameters = parameters_values.shape[0]
		    J_plus = np.zeros((num_parameters, 1))
		    J_minus = np.zeros((num_parameters, 1))
		    gradapprox = np.zeros((num_parameters, 1))

		    # Compute gradapprox
		    for i in range(num_parameters):

		        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
		        # "_" is used because the function you have to outputs two parameters but we only care about the first one
		        ### START CODE HERE ### (approx. 3 lines)
		        thetaplus = np.copy(parameters_values)                                      # Step 1
		        thetaplus[i][0] = thetaplus[i][0] + epsilon                                # Step 2
		        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))                                   # Step 3
		        ### END CODE HERE ###

		        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
		        ### START CODE HERE ### (approx. 3 lines)
		        thetaminus = np.copy(parameters_values)                                     # Step 1
		        thetaminus[i][0] = thetaminus[i][0] - epsilon                               # Step 2
		        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))                                  # Step 3
		        ### END CODE HERE ###

		        # Compute gradapprox[i]
		        ### START CODE HERE ### (approx. 1 line)
		        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
		        ### END CODE HERE ###

		    # Compare gradapprox to backward propagation gradients by computing difference.
		    ### START CODE HERE ### (approx. 1 line)
		    numerator = np.linalg.norm(grad-gradapprox)                                           # Step 1'
		    denominator = np.linalg.norm(grad)+np.linalg.norm(gradapprox)                                         # Step 2'
		    difference = numerator/denominator                                          # Step 3'
		    ### END CODE HERE ###

		    if difference > 2e-7:
		        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
		    else:
		        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

		    return difference
	*/
}

// numericalApprox ...
func numericalApprox() {
	// For checking gradients
}

// RMSdrop ...
func RMSdrop() {
	//LinearWeightedAverage()
	//GradientDescentWithMomentum()
}

// ADAMOptimization is a function
func ADAMOptimization(parameters map[string]float64) (map[string]float64, map[string]float64) {
	//BiasCorrection()
	/*
	   def initialize_adam(parameters) :
	       """
	       Initializes v and s as two python dictionaries with:
	                   - keys: "dW1", "db1", ..., "dWL", "dbL"
	                   - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

	       Arguments:
	       parameters -- python dictionary containing your parameters.
	                       parameters["W" + str(l)] = Wl
	                       parameters["b" + str(l)] = bl

	       Returns:
	       v -- python dictionary that will contain the exponentially weighted average of the gradient.
	                       v["dW" + str(l)] = ...
	                       v["db" + str(l)] = ...
	       s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
	                       s["dW" + str(l)] = ...
	                       s["db" + str(l)] = ...

	       """

	       L = len(parameters) // 2 # number of layers in the neural networks
	       v = {}
	       s = {}

	       # Initialize v, s. Input: "parameters". Outputs: "v, s".
	       for l in range(L):
	       ### START CODE HERE ### (approx. 4 lines)
	           v["dW" + str(l+1)] = np.zeros((parameters['W' + str(l+1)].shape[0],parameters['W' + str(l+1)].shape[1]))
	           v["db" + str(l+1)] = np.zeros((parameters['b' + str(l+1)].shape[0],parameters['b' + str(l+1)].shape[1]))
	           s["dW" + str(l+1)] = np.zeros((parameters['W' + str(l+1)].shape[0],parameters['W' + str(l+1)].shape[1]))
	           s["db" + str(l+1)] = np.zeros((parameters['b' + str(l+1)].shape[0],parameters['b' + str(l+1)].shape[1]))
	       ### END CODE HERE ###

	       return v, s
	*/
	/*
	   def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
	                                   beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
	       """
	       Update parameters using Adam

	       Arguments:
	       parameters -- python dictionary containing your parameters:
	                       parameters['W' + str(l)] = Wl
	                       parameters['b' + str(l)] = bl
	       grads -- python dictionary containing your gradients for each parameters:
	                       grads['dW' + str(l)] = dWl
	                       grads['db' + str(l)] = dbl
	       v -- Adam variable, moving average of the first gradient, python dictionary
	       s -- Adam variable, moving average of the squared gradient, python dictionary
	       learning_rate -- the learning rate, scalar.
	       beta1 -- Exponential decay hyperparameter for the first moment estimates
	       beta2 -- Exponential decay hyperparameter for the second moment estimates
	       epsilon -- hyperparameter preventing division by zero in Adam updates

	       Returns:
	       parameters -- python dictionary containing your updated parameters
	       v -- Adam variable, moving average of the first gradient, python dictionary
	       s -- Adam variable, moving average of the squared gradient, python dictionary
	       """

	       L = len(parameters) // 2                 # number of layers in the neural networks
	       v_corrected = {}                         # Initializing first moment estimate, python dictionary
	       s_corrected = {}                         # Initializing second moment estimate, python dictionary

	       # Perform Adam update on all parameters
	       for l in range(L):
	           # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
	           ### START CODE HERE ### (approx. 2 lines)
	           v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * grads['dW' + str(l+1)]
	           v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * grads['db' + str(l+1)]
	           ### END CODE HERE ###

	           # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
	           ### START CODE HERE ### (approx. 2 lines)
	           v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-pow(beta1,t))
	           v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-pow(beta1,t))
	           ### END CODE HERE ###

	           # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
	           ### START CODE HERE ### (approx. 2 lines)
	           s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * pow(grads['dW' + str(l+1)],2)
	           s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * pow(grads['db' + str(l+1)],2)
	           ### END CODE HERE ###

	           # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
	           ### START CODE HERE ### (approx. 2 lines)
	           s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-pow(beta2,t))
	           s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-pow(beta2,t))
	           ### END CODE HERE ###

	           # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
	           ### START CODE HERE ### (approx. 2 lines)
	           parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon))
	           parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon))
	           ### END CODE HERE ###

	       return parameters, v, s
	*/
	var result map[string]float64
	result = make(map[string]float64)
	return result, result
}

func batch_normalization() {

}

func normalize_inputs() {
	// Mean and variance
	// Not normalizing inputs may lead to poor performance because a small learning rate for gradient descent is required
}
