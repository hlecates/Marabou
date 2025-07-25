#include "TorchConstantBounded.h"

namespace NLR {

TorchConstantBounded::TorchConstantBounded(const torch::Tensor& constant_value) 
    : _constant_value(constant_value),
      _input_size(0),  // Constants don't have inputs
      _output_size(constant_value.numel()),
      _bounds_computed(false),
      _work1_lower(nullptr),
      _work1_upper(nullptr),
      _work2_lower(nullptr),
      _work2_upper(nullptr) {
}

// Standard PyTorch forward pass
torch::Tensor TorchConstantBounded::forward(const torch::Tensor& input) {
    (void)input; // Suppress unused parameter warning
    // Constants ignore input and always return the same value
    return _constant_value;
}

// Auto-LiRPA style boundBackward method (NEW)
// Constants don't depend on inputs, so A matrices are zero
std::tuple<Vector<std::pair<torch::Tensor, torch::Tensor>>, torch::Tensor, torch::Tensor>
TorchConstantBounded::boundBackward(const torch::Tensor& last_lA, 
                                     const torch::Tensor& last_uA,
                                     const Vector<BoundedTensor<torch::Tensor>>& input_bounds) {
    
    (void)input_bounds; // Suppress unused parameter warning
    
    // Constants don't depend on inputs, so A matrices are zero
    // The bias term contains the constant value contribution
    torch::Tensor zero_lA = torch::zeros_like(last_lA);
    torch::Tensor zero_uA = torch::zeros_like(last_uA);
    
    Vector<std::pair<torch::Tensor, torch::Tensor>> A_matrices;
    A_matrices.append(std::make_pair(zero_lA, zero_uA));
    
    // Bias terms contain the constant value contribution
    torch::Tensor lbias = torch::matmul(last_lA, _constant_value);
    torch::Tensor ubias = torch::matmul(last_uA, _constant_value);
    
    return std::make_tuple(A_matrices, lbias, ubias);
}

// CROWN Backward Mode: Propagate bounds backward through Constant
LinearBound TorchConstantBounded::computeCrownBackwardPropagation(const torch::Tensor& lastLowerAlpha, 
                                                                  const torch::Tensor& lastUpperAlpha,
                                                                  const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputBounds) {
    (void)inputBounds; // Suppress unused parameter warning
    // Constant layer has no dependency on input, so alpha matrices become zero
    torch::Tensor zeroAlpha = torch::zeros_like(lastLowerAlpha);
    return LinearBound(zeroAlpha, torch::zeros({lastLowerAlpha.size(0)}), 
                      zeroAlpha, torch::zeros({lastUpperAlpha.size(0)}));
}

// IBP (Interval Bound Propagation): Fast interval-based bound computation for Constant
std::pair<torch::Tensor, torch::Tensor> TorchConstantBounded::computeIntervalBoundPropagation(
    const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputBounds) {
    (void)inputBounds; // Suppress unused parameter warning
    
    // Constant layer always outputs the same value regardless of input
    torch::Tensor constantValue = _constant_value;
    
    // Return the constant value as both lower and upper bound
    return std::make_pair(constantValue, constantValue);
}

// Bound state management
torch::Tensor TorchConstantBounded::getLowerBound() const {
    if (!_bounds_computed) {
        throw std::runtime_error("Bounds not computed yet");
    }
    return _lower_bound;
}

torch::Tensor TorchConstantBounded::getUpperBound() const {
    if (!_bounds_computed) {
        throw std::runtime_error("Bounds not computed yet");
    }
    return _upper_bound;
}

// Working memory management
void TorchConstantBounded::setWorkingMemory(torch::Tensor* workingMemoryLower1,
                                           torch::Tensor* workingMemoryUpper1,
                                           torch::Tensor* workingMemoryLower2,
                                           torch::Tensor* workingMemoryUpper2) {
    _work1_lower = workingMemoryLower1;
    _work1_upper = workingMemoryUpper1;
    _work2_lower = workingMemoryLower2;
    _work2_upper = workingMemoryUpper2;
}

} // namespace NLR 