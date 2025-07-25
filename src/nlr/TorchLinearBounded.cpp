// Marabou/src/nlr/bounded_modules/TorchLinearModule.cpp
#include "TorchLinearBounded.h"

namespace NLR {

TorchLinearModule::TorchLinearModule(const torch::nn::Linear& linearModule, float alpha) 
    : _linear_module(linearModule),
      _alpha(alpha),
      _input_size(linearModule->weight.size(1)),  // Set from weight dimensions
      _output_size(linearModule->weight.size(0)), // Set from weight dimensions
      _bounds_computed(false),
      _work1_lower(nullptr),
      _work1_upper(nullptr),
      _work2_lower(nullptr),
      _work2_upper(nullptr) {
    
    // Register the linear module as a submodule
    register_module("linear", linearModule);
    
    // NOTE: weights and bias are already preprocessed during ONNX conversion in convertGemm()
}

// Standard PyTorch forward pass (inherited from torch::nn::Module)
torch::Tensor TorchLinearModule::forward(const torch::Tensor& input) {
    // Apply matrix multiplication using the linear module's weights
    // (weights are already preprocessed during ONNX conversion)
    torch::Tensor matmul_result = torch::matmul(input, _linear_module->weight.t());
    
    // Apply alpha scaling to matrix multiplication result
    torch::Tensor alpha_scaled = _alpha * matmul_result;
    
    // Add bias (already preprocessed during ONNX conversion)
    torch::Tensor result = alpha_scaled + (_linear_module->bias.defined() ? _linear_module->bias : torch::zeros_like(alpha_scaled[0]));
    
    return result;
}

std::tuple<Vector<std::pair<torch::Tensor, torch::Tensor>>, torch::Tensor, torch::Tensor>
TorchLinearModule::boundBackward(const torch::Tensor& last_lA, 
                                  const torch::Tensor& last_uA,
                                  const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if ( inputBounds.size() < 1 )
    {
        throw std::runtime_error("TorchLinearModule expects at least one input.");
    }

    // extract the weights
    auto weight = _linear_module->weight;
    auto bias = _linear_module->bias;

    // Following auto lirpa just muttiply the weight matrcies
    // next_A = last_A.matmul(weight)
    torch::Tensor lA = last_lA.defined() ? torch::matmul(last_lA, weight) : torch::Tensor();
    torch::Tensor uA = last_lA.defined() ? torch::matmul(last_uA, weight) : torch::Tensor();

    // Bias: new bias = last_A.matmul(bias)
    // Just scaling the old bias
    torch::Tensor lBias = last_lA.defined() ? torch::matmul(last_lA, bias) : torch::zeros({1});
    torch::Tensor uBias = last_lA.defined() ? torch::matmul(last_uA, bias) : torch::zeros({1});

    // Construct the return object, matchs the auto lirpa style
    Vector<std::pair<torch::Tensor, torch::Tensor>> A_matrices;
    A_matrices.append(std::make_pair(lA, uA));

    return std::make_tuple(A_matrices, lBias, uBias);

}


// IBP (Interval Bound Propagation)
std::pair<torch::Tensor, torch::Tensor> TorchLinearModule::computeIntervalBoundPropagation(
    const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputBounds) {
        
    if (inputBounds.size() < 1) {
        throw std::runtime_error("TorchLinearModule expects at least one input");
    }

    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.first;
    torch::Tensor inputUpperBound = inputBoundsPair.second;

    torch::Tensor lowerBound = computeLinearIBPLowerBound(inputLowerBound, inputUpperBound);
    torch::Tensor upperBound = computeLinearIBPUpperBound(inputLowerBound, inputUpperBound);

    _lower_bound = lowerBound;
    _upper_bound = upperBound;
    _bounds_computed = true;

    return std::make_pair(lowerBound, upperBound);
}

// IBP Lower Bound Computation: Compute lower bound using sign-based analysis
torch::Tensor TorchLinearModule::computeLinearIBPLowerBound(
    const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound) {
    
    auto weight = _linear_module->weight;
    auto bias = _linear_module->bias;

    torch::Tensor lowerBound = torch::zeros({weight.size(0)});

    // Note use int64 since torch API generally expects and return them
    for (int64_t i = 0; i < weight.size(0); ++i) {
        torch::Tensor weightI = weight[i];
        float biasI = bias.defined() ? bias[i].item<float>() : 0.0f;

        // For linear layers: f(x) = W*x + b
        // Lower bound: sum over the positive weights * lower input + negatuve weights * upper input
        // The lower bound occurs when we use the lower bound for pos weights and upper bound for neg weights

        // Boolean mask: if value in weightI > 0 == true, then converted to float tensor where true = 1.0, false = 0.0
        torch::Tensor posMask = (weightI > 0).to(torch::kFloat);
        // Equivalent boolean mask for negative weights
        torch::Tensor negMask = (weightI < 0).to(torch::kFloat);
        
        lowerBound[i] = (weightI * posMask * inputLowerBound + weightI * negMask * inputUpperBound).sum() + biasI;
    }

    return lowerBound;

}

// IBP Upper Bound Computation: Compute upper bound using sign-based analysis
torch::Tensor TorchLinearModule::computeLinearIBPUpperBound(
    const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound) {
    
    auto weight = _linear_module->weight;
    auto bias = _linear_module->bias;

    torch::Tensor upperBound = torch::zeros({weight.size(0)});

    for(int64_t i = 0; i < weight.size(0); ++i){
        torch::Tensor weightI = weight[i];
        float biasI = bias.defined() ? bias[i].item<float>() : 0.0f;

        // For linear layers f(x) = W*x + b, the upper bound is computed as:
        // sum(positive_weights * upper_input + negative_weights * lower_input) + bias

        // Boolean mask: if value in weightI > 0 == true, then converted to float tensor where true = 1.0, false = 0.0
        torch::Tensor posMask = (weightI > 0).to(torch::kFloat);
        // Equivalent boolean mask for negative weights
        torch::Tensor negMask = (weightI < 0).to(torch::kFloat);

        upperBound[i] = (weightI * negMask * inputLowerBound + weightI * posMask * inputUpperBound).sum() + biasI;

    }

    return upperBound;
}
    

void TorchLinearModule::setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) {
    _neuronToMarabouMap = neuronMapping;
}

const Map<unsigned, Vector<Variable>>& TorchLinearModule::getNeuronToMarabouMap() const {
    return _neuronToMarabouMap;
}

// Bound state management
torch::Tensor TorchLinearModule::getLowerBound() const {
    if (!_bounds_computed) {
        throw std::runtime_error("Bounds not computed");
    }
    return _lower_bound;
}

torch::Tensor TorchLinearModule::getUpperBound() const {
    if (!_bounds_computed) {
        throw std::runtime_error("Bounds not computed");
    }
    return _upper_bound;
}

// Working memory management for efficient bound computation
void TorchLinearModule::setWorkingMemory(torch::Tensor* workingMemoryLower1,
                                          torch::Tensor* workingMemoryUpper1,
                                          torch::Tensor* workingMemoryLower2,
                                          torch::Tensor* workingMemoryUpper2) {
    _work1_lower = workingMemoryLower1;
    _work1_upper = workingMemoryUpper1;
    _work2_lower = workingMemoryLower2;
    _work2_upper = workingMemoryUpper2;
}

} // namespace NLR